"""Merge contracts: independent learned-position oracle, weighted losses and empty updates."""

import copy
import unittest
from dataclasses import asdict, replace

import torch

from examples.post_training.distillation import distillation_loss
from examples.principles.decoder import ReferenceDecoder
from llms_from_scratch import (
    BlockConfig,
    ModelConfig,
    Transformer,
    basic_decoder_config,
    token_loss_sum,
)
from llms_from_scratch.analysis import ffn_cost
from llms_from_scratch.inference.sampling import distribution, sample_generate
from llms_from_scratch.layers import FeedForward


class CurriculumTests(unittest.TestCase):
    def test_sampling_rejects_empty_vocabulary_and_noninteger_ids(self):
        logits = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float64).log()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            distribution(torch.empty(0))
        for history in ([0.5], [True], [-1], [4]):
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, "history IDs"):
                distribution(logits, history=history)
        with self.assertRaisesRegex(ValueError, "top_k"):
            distribution(logits, top_k=True)
        torch.testing.assert_close(
            distribution(logits, top_k=2),
            torch.tensor([0.0, 4 / 7, 3 / 7, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            distribution(logits, history=iter([0, 1, 1]), repetition_penalty=2),
            torch.tensor([0.01, 0.16, 0.3, 0.2], dtype=torch.float64) / 0.67,
        )

    def test_distillation_one_hot_float32_and_ignored_nonfinite_rows(self):
        student = torch.tensor([[0.1, 0.3, -0.2], [float("nan")] * 3], requires_grad=True)
        teacher = torch.tensor(
            [[0.0, -torch.inf, -torch.inf], [float("nan")] * 3], requires_grad=True
        )
        valid = torch.tensor([True, False])
        actual = distillation_loss(student, teacher, valid)
        expected = torch.nn.functional.cross_entropy(student[:1], torch.tensor([0]))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        torch.testing.assert_close(student.grad[1], torch.zeros(3))
        self.assertIsNone(teacher.grad)
        student.grad = None
        distillation_loss(student, teacher, torch.zeros_like(valid)).backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        for invalid in (float("nan"), torch.inf, -torch.inf):
            with self.subTest(student=invalid), self.assertRaisesRegex(ValueError, "student"):
                distillation_loss(torch.full((1, 3), invalid), teacher[:1], valid[:1])
        for values in ([float("nan"), 0.0, 0.0], [torch.inf, 0.0, 0.0], [-torch.inf] * 3):
            with self.subTest(teacher=values), self.assertRaisesRegex(ValueError, "teacher"):
                distillation_loss(student[:1], torch.tensor([values]), valid[:1])
        with self.assertRaisesRegex(ValueError, "temperature scaling"):
            distillation_loss(student[:1], teacher[:1], valid[:1], temperature=1e-300)

    def test_generation_rejects_ignored_or_conflicting_encoder_sources(self):
        config = basic_decoder_config(8, dim=4, heads=2, ff_dim=8, layers=1)
        ids = torch.tensor([[0, 1]])
        decoder = Transformer(config)
        with self.assertRaisesRegex(ValueError, "only used by encoder-decoder"):
            decoder.generate(ids, 1, source_ids=ids)
        seq2seq = Transformer(replace(config, architecture="encoder_decoder"))
        memory = seq2seq.encode(ids)
        for length in (0, 1):
            with self.assertRaisesRegex(ValueError, "not both"):
                seq2seq.generate(ids, length, source_ids=ids, memory=memory)
            with self.assertRaisesRegex(ValueError, "already contains"):
                seq2seq.generate(
                    ids, length, memory=memory, source_valid=torch.ones_like(ids).bool()
                )
        self.assertTrue(seq2seq.training)

    def test_learned_baseline_against_independent_reference(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        reference = ReferenceDecoder(
            8, width=4, heads=2, ff_width=8, layers=1, max_length=6
        ).double()
        model = Transformer(
            basic_decoder_config(8, dim=4, heads=2, ff_dim=8, layers=1, max_length=6)
        ).double()
        with torch.no_grad():
            model.embedding.load_state_dict(reference.embedding.state_dict())
            model.position.load_state_dict(reference.position.state_dict())
            model.head.load_state_dict(reference.head.state_dict())
            model.norm.load_state_dict(reference.norm.state_dict())
            for block, old in zip(model.blocks, reference.blocks, strict=True):
                block.norms[0].load_state_dict(old.norm_attention.state_dict())
                block.norms[1].load_state_dict(old.norm_ff.state_dict())
                block.ff.up.load_state_dict(old.ff[0].state_dict())
                block.ff.down.load_state_dict(old.ff[2].state_dict())
                block.attention.output.load_state_dict(old.attention.output.state_dict())
                for index, projection in enumerate(
                    (block.attention.q, block.attention.k, block.attention.v)
                ):
                    projection.weight.copy_(old.attention.qkv.weight.chunk(3, 0)[index])
                    projection.bias.copy_(old.attention.qkv.bias.chunk(3, 0)[index])
        ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        actual, expected = model(ids).logits, reference(ids)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(model.embedding.weight.grad, reference.embedding.weight.grad)
        torch.testing.assert_close(model.position.weight.grad, reference.position.weight.grad)
        torch.testing.assert_close(
            model.blocks[0].attention.v.weight.grad,
            reference.blocks[0].attention.qkv.weight.grad.chunk(3, 0)[2],
        )
        cache, parts, start = None, [], 0
        for length in (2, 1, 2):
            output = model(ids[:, start : start + length], cache=cache, use_cache=True)
            parts.append(output.logits)
            cache, start = output.cache, start + length
        torch.testing.assert_close(torch.cat(parts, 1), expected)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 276)
        self.assertEqual(ModelConfig.from_dict(asdict(model.config)), model.config)
        prompt = ids[:1, :2]
        cached = sample_generate(model, prompt, 3, generator=torch.Generator().manual_seed(9))
        fresh = sample_generate(
            model, prompt, 3, cached=False, generator=torch.Generator().manual_seed(9)
        )
        self.assertTrue(torch.equal(cached, fresh))
        self.assertTrue(model.training)

    def test_biased_ffn_parameters_are_not_matmul_weights(self):
        config = BlockConfig(activation="relu", ff_dim=8, bias=True)
        layer = FeedForward(4, 8, activation="relu", bias=True)
        costs = ffn_cost(4, config, tokens=3)
        self.assertEqual(costs["total_parameters"], sum(p.numel() for p in layer.parameters()))
        self.assertEqual(costs["total_parameters"], 76)
        self.assertEqual(costs["matmul_flops"], 3 * 2 * 64)
        with self.assertRaises(ValueError):
            basic_decoder_config(8, heads=0)

    def test_sum_count_aggregation_and_empty_adamw_update(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 1, 2], [3, 2, -999]])
        valid = targets != -999
        complete = token_loss_sum(logits, targets, valid)
        parts = [
            token_loss_sum(logits[i : i + 1], targets[i : i + 1], valid[i : i + 1])
            for i in range(2)
        ]
        torch.testing.assert_close(
            sum(p.loss_sum for p in parts) / sum(p.valid_count for p in parts), complete.mean()
        )
        self.assertEqual(complete.valid_count.item(), 5)
        empty = token_loss_sum(logits, targets, torch.zeros_like(valid))
        empty.loss_sum.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        with self.assertRaises(ValueError):
            empty.mean()
        model = Transformer(basic_decoder_config(4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        before = copy.deepcopy(model.state_dict())
        state = copy.deepcopy(optimizer.state_dict())
        result = token_loss_sum(
            model(torch.tensor([[0, 1]])).logits,
            torch.tensor([[1, 2]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        if result.valid_count.item():
            result.mean().backward()
            optimizer.step()
        self.assertEqual(optimizer.state_dict(), state)
        for key, parameter in before.items():
            torch.testing.assert_close(parameter, model.state_dict()[key], rtol=0, atol=0)
        # DDP averages gradients: each rank scales its local sum by world_size/global_count.
        full_grad = torch.autograd.grad(complete.mean(), logits)[0]
        ddp_objectives = [p.loss_sum * 2 / complete.valid_count for p in parts]
        averaged_grad = (
            sum(torch.autograd.grad(x, logits, retain_graph=True)[0] for x in ddp_objectives) / 2
        )
        torch.testing.assert_close(full_grad, averaged_grad)


if __name__ == "__main__":
    unittest.main()
