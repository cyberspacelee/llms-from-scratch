import torch
import torch.nn.functional as F

from llms_from_scratch.post_training.chat_template import IGNORE_INDEX, encode_chat
from llms_from_scratch.post_training.common import tiny_gpt
from llms_from_scratch.post_training.sft import (
    document_causal_mask,
    pack_examples,
    packed_logits,
    sft_loss,
    train_sft,
)
from llms_from_scratch.post_training.tasks import addition_problems, chat_tokenizer


def _examples(tok):
    return [
        encode_chat(tok, [{"role": "user", "content": p.prompt + "?"},
                          {"role": "assistant", "content": p.answer}])
        for p in addition_problems(4)
    ]


def test_sft_loss_matches_manual_masked_mean():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, IGNORE_INDEX],
                           [IGNORE_INDEX, 1, 2, IGNORE_INDEX, IGNORE_INDEX]])
    # 被监督的目标 (b, t) 由位置 t-1 的 logits 预测
    pairs = [(0, 2), (0, 3), (1, 1), (1, 2)]
    terms = [F.cross_entropy(logits[b, t - 1], labels[b, t]) for b, t in pairs]
    torch.testing.assert_close(sft_loss(logits, labels), torch.stack(terms).mean())


def test_masked_positions_receive_no_gradient():
    logits = torch.randn(1, 4, 6, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 3]])
    sft_loss(logits, labels).backward()
    assert logits.grad[0, 0].abs().sum() == 0  # 位置 0 预测的是被掩码的 token 1
    assert logits.grad[0, 3].abs().sum() == 0  # 最后一个位置没有下一个 token
    assert logits.grad[0, 1].abs().sum() > 0


def test_document_mask_is_block_diagonal_causal():
    doc_ids = torch.tensor([[0, 0, 1, 1, 1, -1]])
    m = document_causal_mask(doc_ids)[0]
    assert m[1, 0] and not m[0, 1]
    assert not m[2, 1] and m[4, 2]
    assert m[5, 5] and not m[5, 4]


def test_packed_forward_equals_separate_forward():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size, seed=1).eval()
    examples = _examples(tok)[:5]
    batch = pack_examples(examples, max_len=64, pad_id=tok.pad_id)
    assert batch.input_ids.shape[0] == 3  # 每条 24 个 token，一行装两条
    logits = packed_logits(model, batch)
    for r in range(batch.input_ids.shape[0]):
        for d in range(int(batch.doc_ids[r].max()) + 1):
            pos = (batch.doc_ids[r] == d).nonzero().squeeze(1)
            alone = model(batch.input_ids[r, pos][None])[0]
            torch.testing.assert_close(logits[r, pos], alone, atol=1e-5, rtol=1e-4)
    # 每个文档的第一个标签被屏蔽：不跨文档预测
    starts = (batch.position_ids == 0) & (batch.doc_ids >= 0)
    assert (batch.labels[starts] == IGNORE_INDEX).all()


def test_without_document_mask_packing_leaks_context():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size, seed=1).eval()
    batch = pack_examples(_examples(tok)[:2], max_len=64, pad_id=tok.pad_id)
    second = (batch.doc_ids[0] == 1).nonzero().squeeze(1)
    naive = model(batch.input_ids)[0, second]  # 普通因果掩码：第二条能看到第一条
    alone = model(batch.input_ids[0, second][None])[0]
    assert (naive - alone).abs().max() > 1e-3


def test_sft_learns_chat_format():
    torch.set_num_threads(1)
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size, seed=0)
    losses = train_sft(model, _examples(tok), tok.pad_id, steps=60, lr=1e-2)
    assert sum(losses[-5:]) / 5 < 0.5 * sum(losses[:5]) / 5
