"""CPU invariants with real forwards/backwards, not tests mirroring implementation."""

import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from transformer_lab import (
    AttentionConfig,
    BlockConfig,
    ModelConfig,
    PositionConfig,
    Transformer,
    gathered_attention,
    language_model_loss,
    make_attention,
    next_token_loss,
    scaled_dot_product_attention,
    teacher_forcing,
    token_loss,
)
from transformer_lab.analysis import attention_cost, ffn_cost, parameter_count
from transformer_lab.cache import KVCache, ModelCache, PagedKVCache, QuantizedTensor
from transformer_lab.inference import compressed_causal_attention, greedy_speculative_generate
from transformer_lab.layers import FeedForward, LayerNorm, MixtureOfExperts, RMSNorm
from transformer_lab.position import apply_rope, sinusoidal
from transformer_lab.presets import PRESETS, preset


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(19)
        self.ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]])

    def assert_close(
        self, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-9, rtol: float = 1e-7
    ) -> None:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def cached_parts(
        self,
        model: Transformer,
        ids: torch.Tensor,
        chunks: tuple[int, ...] = (2, 1, 4),
        source: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ModelCache]:
        state, output, offset = None, [], 0
        for size in chunks:
            part = model(
                ids[:, offset : offset + size],
                source_ids=source if state is None else None,
                valid=None if valid is None else valid[:, offset : offset + size],
                cache=state,
                use_cache=True,
            )
            state = part.cache
            self.assertEqual(state.length, offset + size)
            self.assertTrue(
                all(layer.self_attention.length == state.length for layer in state.layers)
            )
            output.append(part.logits)
            offset += size
        return torch.cat(output, 1), state

    def test_attention_hand_case_and_sdpa_gradients(self) -> None:
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
        k = q.clone()
        v = torch.tensor([[[[2.0, 3.0], [4.0, 7.0]]]], dtype=q.dtype)
        mask = torch.ones(2, 2, dtype=torch.bool).tril()
        actual = scaled_dot_product_attention(q, k, v, mask)
        p = torch.sigmoid(torch.tensor(2**-0.5, dtype=q.dtype))
        self.assert_close(actual[0, 0, 0], v[0, 0, 0])
        self.assert_close(actual[0, 0, 1], (1 - p) * v[0, 0, 0] + p * v[0, 0, 1])
        for kind in ("mha", "mqa", "gqa"):
            for cross in (False, True):
                with self.subTest(kind=kind, cross=cross):
                    c = AttentionConfig(kind=kind, qk_norm=True)
                    manual = make_attention(32, c, cross=cross).double()
                    sdpa = make_attention(32, replace(c, backend="sdpa"), cross=cross).double()
                    sdpa.load_state_dict(manual.state_dict())
                    x = torch.randn(2, 3, 32, dtype=torch.float64, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    memory = torch.randn(2, 5, 32, dtype=torch.float64) if cross else None
                    mask = torch.tensor(
                        [[True] * (5 if cross else 3), [False] * (5 if cross else 3)]
                    )
                    a = manual(x, memory=memory, causal=not cross, key_valid=mask)[0]
                    b = sdpa(y, memory=memory, causal=not cross, key_valid=mask)[0]
                    self.assert_close(a, b)
                    a.square().sum().backward()
                    b.square().sum().backward()
                    self.assert_close(x.grad, y.grad)
                    for pa, pb in zip(manual.parameters(), sdpa.parameters(), strict=True):
                        self.assert_close(pa.grad, pb.grad)
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_group_mapping_and_cache_memory(self) -> None:
        for kind in ("mha", "mqa", "gqa"):
            c = AttentionConfig(kind=kind, position=PositionConfig(kind="none"))
            module = make_attention(32, c).double()
            x = torch.randn(2, 4, 32, dtype=torch.float64)
            y, cache = module(x, causal=True, use_cache=True)
            q = module.q(x).reshape(2, 4, 4, 8).transpose(1, 2)
            outputs = []
            for h in range(4):
                group = h // (4 // c.effective_kv_heads)
                output = scaled_dot_product_attention(
                    q[:, h : h + 1],
                    cache.key[:, group : group + 1],
                    cache.value[:, group : group + 1],
                    torch.ones(4, 4, dtype=torch.bool).tril(),
                )
                outputs.append(output)
            reference = module.output(torch.cat(outputs, 1).transpose(1, 2).reshape(2, 4, 32))
            self.assert_close(y, reference)
            cost = attention_cost(c, 32, batch=2, query_tokens=4, key_tokens=4, bytes_per_element=8)
            self.assertEqual(cache.nbytes, cost.cache_bytes)

    def test_position_invariants_and_scaling(self) -> None:
        positions = torch.arange(7)
        q, k = [torch.randn(2, 4, 7, 8, dtype=torch.float64) for _ in range(2)]
        for scaling in ("none", "linear", "ntk", "yarn"):
            c = PositionConfig(scaling=scaling, factor=4)
            qr, kr = apply_rope(q, positions, c), apply_rope(k, positions, c)
            self.assert_close(qr.square().sum(-1), q.square().sum(-1))
            self.assert_close(
                qr @ kr.transpose(-1, -2),
                apply_rope(q, positions + 20, c)
                @ apply_rope(k, positions + 20, c).transpose(-1, -2),
            )
        linear = apply_rope(q, positions, PositionConfig(scaling="linear", factor=2))
        self.assert_close(linear, apply_rope(q, positions.double() / 2))
        self.assertEqual(sinusoidal(positions, 7, torch.float64).shape, (7, 7))
        self.assert_close(
            sinusoidal(torch.tensor([0]), 4).double(),
            torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float64),
        )

    def test_mla_absorbed_naive_outputs_gradients_and_cross_cache(self) -> None:
        for cross in (False, True):
            for rank in (0, 8):
                with self.subTest(cross=cross, rank=rank):
                    c = AttentionConfig(kind="mla", q_rank=rank, mla_impl="naive")
                    naive = make_attention(32, c, cross=cross).double()
                    absorbed = make_attention(
                        32, replace(c, mla_impl="absorbed"), cross=cross
                    ).double()
                    absorbed.load_state_dict(naive.state_dict())
                    x = torch.randn(2, 4, 32, dtype=torch.float64, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    memory = torch.randn(2, 5, 32, dtype=torch.float64) if cross else None
                    a, cache = naive(x, memory=memory, causal=not cross, use_cache=True)
                    b, state = absorbed(y, memory=memory, causal=not cross, use_cache=True)
                    self.assert_close(a, b)
                    a.square().mean().backward()
                    b.square().mean().backward()
                    self.assert_close(x.grad, y.grad)
                    for pa, pb in zip(naive.parameters(), absorbed.parameters(), strict=True):
                        self.assert_close(pa.grad, pb.grad)
                    self.assertEqual(
                        cache.nbytes, 2 * (5 if cross else 4) * (c.kv_rank + c.rope_dim) * 8
                    )
                    if cross:
                        continued, reused = absorbed(
                            x[:, :1], cache=state, query_offset=4, use_cache=True
                        )
                        full, _ = absorbed(x[:, :1], memory=memory, query_offset=4)
                        self.assert_close(continued, full)
                        self.assertIs(reused, state)

    def test_cached_full_matrix(self) -> None:
        configs = [preset(name) for name in PRESETS]
        configs += [
            ModelConfig(block=BlockConfig(attention=AttentionConfig(kind=kind)))
            for kind in ("mha", "mqa")
        ]
        for pattern in ("sliding", "local", "block_sparse", "token_sparse"):
            for kind in ("gqa", "mla"):
                configs.append(
                    ModelConfig(
                        block=BlockConfig(
                            attention=AttentionConfig(kind=kind, pattern=pattern, window=3)
                        )
                    )
                )
        for scaling in ("linear", "ntk", "yarn"):
            configs.append(
                ModelConfig(
                    block=BlockConfig(
                        attention=AttentionConfig(
                            position=PositionConfig(scaling=scaling, factor=4)
                        )
                    )
                )
            )
        for config in configs:
            with self.subTest(config=config):
                model = Transformer(config).double().eval()
                source = self.ids[:, :5] if config.architecture == "encoder_decoder" else None
                full = model(self.ids, source_ids=source).logits
                changed = self.ids.clone()
                changed[:, 4:] = 12
                self.assert_close(full[:, :4], model(changed, source_ids=source).logits[:, :4])
                for chunks in ((7,), (1,) * 7, (2, 1, 4)):
                    actual, state = self.cached_parts(model, self.ids, chunks, source)
                    self.assert_close(full, actual)
                if source is not None:
                    self.assertTrue(all(layer.cross_attention.static for layer in state.layers))
                output = model(self.ids, source_ids=source)
                (output.logits.square().mean() + 0.01 * output.auxiliary_loss).backward()
                self.assertTrue(
                    all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                )

    def test_padding_and_batch_isolation(self) -> None:
        valid = torch.tensor([[False, True, True, True, False, False, False], [True] * 7])
        for name in ("llama", "deepseek", "qwen_hybrid", "classic"):
            model = Transformer(preset(name)).double().eval()
            source = self.ids if name == "classic" else None
            source_valid = valid if source is not None else None
            full = model(self.ids, valid=valid, source_ids=source, source_valid=source_valid).logits
            changed = self.ids.clone()
            changed[~valid] = 15
            actual = model(
                changed,
                valid=valid,
                source_ids=changed if source is not None else None,
                source_valid=source_valid,
            ).logits
            self.assert_close(full[valid], actual[valid])
            self.assertTrue(torch.isfinite(full).all())
            for batch in (0, 1):
                isolated = model(
                    self.ids[batch : batch + 1],
                    valid=valid[batch : batch + 1],
                    source_ids=None if source is None else source[batch : batch + 1],
                    source_valid=None if source_valid is None else source_valid[batch : batch + 1],
                ).logits
                self.assert_close(full[batch : batch + 1], isolated)
            cache, parts, offset = None, [], 0
            for size in (2, 1, 4):
                output = model(
                    self.ids[:, offset : offset + size],
                    valid=valid[:, offset : offset + size],
                    source_ids=source if cache is None else None,
                    source_valid=source_valid if cache is None else None,
                    cache=cache,
                    use_cache=True,
                )
                parts.append(output.logits)
                cache, offset = output.cache, offset + size
            self.assert_close(full, torch.cat(parts, 1))

    def test_encoder_memory_and_asymmetry(self) -> None:
        config = replace(
            preset("classic"),
            encoder_layers=3,
            encoder_block=replace(preset("llama").block, attention=AttentionConfig(kind="mqa")),
            cross_attention=AttentionConfig(kind="mla"),
        )
        model = Transformer(config).double().eval()
        memory = model.encode(self.ids[:, :5])
        self.assertEqual(len(model.encoder_blocks), 3)
        self.assert_close(
            model(self.ids, source_ids=self.ids[:, :5]).logits,
            model(self.ids, memory=memory).logits,
        )
        with self.assertRaises(ValueError):
            model.encode(self.ids[:, :1], past=memory, use_cache=True)
        config = preset("causal_seq2seq")
        model = Transformer(config).double().eval()
        full = model.encode(self.ids, use_cache=True)
        first = model.encode(self.ids[:, :3], use_cache=True)
        rest = model.encode(self.ids[:, 3:], past=first, use_cache=True)
        self.assert_close(full.hidden, rest.hidden)
        # A changed source suffix cannot affect aligned causal decoder prefixes.
        changed = self.ids.clone()
        changed[:, 4:] = 12
        self.assert_close(
            model(self.ids, source_ids=self.ids).logits[:, :4],
            model(self.ids, source_ids=changed).logits[:, :4],
        )

    def test_encoder_only_is_bidirectional(self) -> None:
        model = Transformer(replace(preset("llama"), architecture="encoder")).double().eval()
        changed = self.ids.clone()
        changed[:, -1] = 12
        self.assertFalse(
            torch.allclose(model(self.ids).logits[:, :1], model(changed).logits[:, :1])
        )
        with self.assertRaises(ValueError):
            model(self.ids, use_cache=True)
        causal = (
            Transformer(replace(preset("llama"), architecture="encoder", encoder_causal=True))
            .double()
            .eval()
        )
        actual, _ = self.cached_parts(causal, self.ids)
        self.assert_close(actual, causal(self.ids).logits)

    def test_block_variants(self) -> None:
        for norm in ("layer", "rms"):
            for order in ("pre", "post"):
                for activation in ("relu", "gelu", "glu", "geglu", "swiglu"):
                    config = ModelConfig(
                        block=BlockConfig(
                            norm=norm, norm_order=order, activation=activation, residual="gated"
                        )
                    )
                    model = Transformer(config).double().eval()
                    actual, _ = self.cached_parts(model, self.ids)
                    self.assert_close(actual, model(self.ids).logits)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware/build unavailable")
    def test_cuda_forward_backward_and_cache(self) -> None:
        device = torch.device("cuda")
        ids = self.ids.to(device)
        dtypes = [torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for name in ("classic", "llama", "deepseek", "qwen_hybrid"):
            for dtype in dtypes:
                with self.subTest(preset=name, dtype=dtype):
                    model = Transformer(preset(name)).to(device=device, dtype=dtype).eval()
                    source = ids[:, :4] if name == "classic" else None
                    full = model(ids, source_ids=source)
                    actual, state = self.cached_parts(model, ids, source=source)
                    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-5
                    self.assert_close(actual, full.logits, atol=tolerance, rtol=tolerance)
                    self.assertEqual(state.valid.device.type, "cuda")
                    (full.logits.float().square().mean() + full.auxiliary_loss).backward()
                    self.assertTrue(
                        all(
                            p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()
                        )
                    )

    def test_norms_ffns_and_ledger(self) -> None:
        x = torch.randn(2, 3, 32, dtype=torch.float64)
        norm = LayerNorm(32).double()
        self.assert_close(norm(x), F.layer_norm(x, (32,), norm.weight, norm.bias, norm.eps))
        rms = RMSNorm(32).double()
        self.assert_close(rms(x), F.rms_norm(x, (32,), rms.weight, rms.eps))
        for activation in ("relu", "gelu", "glu", "geglu", "swiglu"):
            config = BlockConfig(activation=activation)
            ffn = FeedForward(32, 64, activation).double()
            self.assertEqual(parameter_count(ffn), ffn_cost(32, config)["total_parameters"])
            self.assertEqual(ffn(x).shape, x.shape)
        for kind in ("mha", "mqa", "gqa", "mla", "linear", "delta", "gated_delta"):
            c = AttentionConfig(
                kind=kind,
                qk_norm=kind == "gqa",
                position=PositionConfig(kind="none")
                if kind in {"linear", "delta", "gated_delta"}
                else PositionConfig(),
            )
            module = make_attention(32, c).double()
            self.assertEqual(parameter_count(module), attention_cost(c, 32).parameters)
            _, state = module(x, causal=True, use_cache=True)
            self.assertEqual(
                state.nbytes,
                attention_cost(
                    c, 32, batch=2, query_tokens=3, key_tokens=3, bytes_per_element=8
                ).cache_bytes,
            )

    def test_moe_dispatch_and_bias_balance(self) -> None:
        for score in ("softmax", "sigmoid"):
            config = BlockConfig(experts=4, shared_experts=1, router_score=score, balance="bias")
            moe = MixtureOfExperts(32, config).double()
            x = torch.randn(2, 3, 32, dtype=torch.float64, requires_grad=True)
            ids, weights, _ = moe.route(x.reshape(-1, 32))
            dense = torch.stack([e(x.reshape(-1, 32)) for e in moe.experts], 1)
            reference = (
                dense.gather(1, ids[..., None].expand(-1, -1, 32)) * weights[..., None]
            ).sum(1)
            reference += sum(e(x.reshape(-1, 32)) for e in moe.shared)
            y, aux, counts = moe(x)
            self.assert_close(y, reference.reshape_as(x))
            self.assertEqual(int(counts.sum()), 12)
            self.assertEqual(parameter_count(moe), ffn_cost(32, config)["total_parameters"])
            old = moe.selection_bias.clone()
            moe.update_balance(torch.tensor([12, 0, 0, 0]))
            self.assertLess(moe.selection_bias[0], old[0])
            self.assertEqual(aux.item(), 0)
            y.square().sum().backward()
            self.assertTrue(torch.isfinite(moe.router.weight.grad).all())
            self.assertGreater(moe.router.weight.grad.abs().sum(), 0)

    def test_linear_parallel_kernel_and_delta_overwrite(self) -> None:
        c = AttentionConfig(kind="linear", position=PositionConfig(kind="none"))
        module = make_attention(32, c).double()
        x = torch.randn(2, 5, 32, dtype=torch.float64)
        q, k = (
            F.elu(layer(x).reshape(2, 5, 4, 8).transpose(1, 2)) + 1
            for layer in (module.q, module.k)
        )
        v = module.v(x).reshape(2, 5, 4, 8).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2)) * torch.ones(5, 5, dtype=torch.bool).tril()
        y = weights @ v / weights.sum(-1, keepdim=True)
        reference = module.output(y.transpose(1, 2).reshape(2, 5, 32))
        self.assert_close(module(x, causal=True)[0], reference)
        # Isolated delta equation: beta=1 on a unit key overwrites that key's value.
        state = torch.randn(3, 4, dtype=torch.float64)
        key, value = (
            F.normalize(torch.randn(3, dtype=torch.float64), dim=0),
            torch.randn(4, dtype=torch.float64),
        )
        updated = state + key[:, None] * (value - key @ state)[None, :]
        self.assert_close(key @ updated, value)

    def test_sparse_gather_and_compression_causality(self) -> None:
        q, k, v = [torch.randn(2, 4, 7, 8, dtype=torch.float64) for _ in range(3)]
        indices = torch.tensor([[0, 0], [0, 1], [1, 2], [0, 3], [2, 4], [0, 5], [3, 6]])
        valid = torch.ones_like(indices, dtype=torch.bool)
        valid[0, 1] = False
        mask = torch.zeros(7, 7, dtype=torch.bool)
        for row in range(7):
            mask[row, indices[row][valid[row]]] = True
        self.assert_close(
            gathered_attention(q, k, v, indices, valid), scaled_dot_product_attention(q, k, v, mask)
        )
        full = compressed_causal_attention(q, k, v, block_size=2, window=2)
        parts = [
            compressed_causal_attention(
                q[:, :, i : i + 1],
                k[:, :, : i + 1],
                v[:, :, : i + 1],
                block_size=2,
                window=2,
                query_offset=i,
            )
            for i in range(7)
        ]
        self.assert_close(full, torch.cat(parts, -2))
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 4:] *= 10
        changed_v[:, :, 4:] *= 10
        self.assert_close(
            full[:, :, :4],
            compressed_causal_attention(q, changed_k, changed_v, block_size=2, window=2)[:, :, :4],
        )

    def test_teacher_forcing_mtp_alignment_and_gradients(self) -> None:
        tokens = self.ids[:1]
        inputs, targets, valid = teacher_forcing(tokens, 9)
        self.assertEqual(inputs.tolist(), [[9, 0, 1, 2, 3, 4, 5]])
        self.assertTrue(torch.equal(targets, tokens))
        model = Transformer(replace(preset("llama"), mtp_depth=3)).double()
        output = model(tokens)
        predictions, _, _ = model.mtp(output.hidden, tokens, model.embedding, model.head)
        self.assertEqual([p.shape[1] for p in predictions], [5, 4, 3])
        loss, terms, _ = language_model_loss(model, output, tokens)
        self.assert_close(terms["ntp"], token_loss(output.logits[:, :-1], tokens[:, 1:]))
        for depth, prediction in enumerate(predictions, 2):
            self.assert_close(terms[f"mtp_{depth}"], token_loss(prediction, tokens[:, depth:]))
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )
        # Depth 2 can use x(t+1), but must not see its target x(t+2).
        changed = tokens.clone()
        changed[:, 3:] = 15
        altered = model(changed)
        prediction = model.mtp(altered.hidden, changed, model.embedding, model.head)[0][0]
        self.assert_close(predictions[0][:, :2], prediction[:, :2])
        zeros = torch.zeros_like(tokens, dtype=torch.bool)
        loss, _, _ = language_model_loss(model, output, tokens, zeros)
        self.assertEqual(loss.item(), 0)

    def test_pages_prefix_fork_and_quantization(self) -> None:
        k, v = [torch.randn(2, 2, 7, 8, dtype=torch.float64) for _ in range(2)]
        pages = PagedKVCache(3)
        pages.append(k[:, :, :4], v[:, :, :4])
        fork = pages.fork()
        self.assertEqual(pages.block_table, fork.block_table)
        pages.append(k[:, :, 4:], v[:, :, 4:])
        fork.append(k[:, :, 4:5] * 2, v[:, :, 4:5] * 2)
        self.assertEqual(pages.block_table[0], fork.block_table[0])
        self.assertNotEqual(pages.block_table[1], fork.block_table[1])
        self.assert_close(pages.materialize().key, k)
        self.assert_close(fork.materialize().key, torch.cat((k[:, :, :4], k[:, :, 4:5] * 2), -2))
        encoded = QuantizedTensor.encode(k)
        self.assertTrue(((encoded.decode() - k).abs() <= encoded.scales / 2 + 1e-12).all())
        self.assertLess(encoded.nbytes, k.numel() * k.element_size())
        self.assert_close(QuantizedTensor.encode(torch.zeros_like(k)).decode(), torch.zeros_like(k))

    def test_generation_speculation_and_optimizer_step(self) -> None:
        for name in ("classic", "llama", "deepseek", "qwen_hybrid"):
            config = preset(name)
            model = Transformer(config).double().train()
            source = self.ids[:, :4] if config.architecture == "encoder_decoder" else None
            self.assertTrue(
                torch.equal(
                    model.generate(self.ids[:, :2], 4, source_ids=source),
                    model.generate(self.ids[:, :2], 4, source_ids=source, cached=False),
                )
            )
            self.assertTrue(model.training)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            before = model.embedding.weight.detach().clone()
            output = model(self.ids, source_ids=source)
            loss = next_token_loss(output.logits, self.ids) + 0.01 * output.auxiliary_loss
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.embedding.weight))
        target = Transformer(preset("llama")).double()
        prompt = self.ids[:1, :2]
        for draft in (target, Transformer(preset("llama")).double()):
            actual, stats = greedy_speculative_generate(target, draft, prompt, 7, 3)
            self.assertTrue(torch.equal(actual, target.generate(prompt, 7)))
            self.assertGreater(stats["proposed"], 0)

    def test_invalid_configs_inputs_and_cache(self) -> None:
        model = Transformer(preset("llama")).double()
        state = model(self.ids[:, :2], use_cache=True).cache
        wrong = replace(state, layers=state.layers[:1])
        stale = replace(
            state,
            layers=(
                replace(
                    state.layers[0],
                    self_attention=KVCache(
                        state.layers[0].self_attention.key[:, :, :1],
                        state.layers[0].self_attention.value[:, :, :1],
                    ),
                ),
            )
            + state.layers[1:],
        )
        calls = [
            lambda: AttentionConfig(heads=3, kv_heads=2),
            lambda: AttentionConfig(kind="mla", qk_norm=True),
            lambda: PositionConfig(factor=0),
            lambda: BlockConfig(experts=1, top_k=2),
            lambda: ModelConfig(layer_blocks=(BlockConfig(),)),
            lambda: model(torch.tensor([[64]])),
            lambda: model(torch.zeros(1, 2)),
            lambda: model(self.ids, valid=torch.ones_like(self.ids)),
            lambda: model(self.ids, cache=state),
            lambda: model(self.ids, cache=wrong, use_cache=True),
            lambda: model(self.ids, cache=stale, use_cache=True),
            lambda: model.generate(self.ids, 300),
            lambda: model.generate(self.ids, 1, eos_id=64),
            lambda: gathered_attention(
                torch.zeros(1, 1, 1, 2),
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 2, 2),
                torch.tensor([[0, 0]]),
            ),
        ]
        for call in calls:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
