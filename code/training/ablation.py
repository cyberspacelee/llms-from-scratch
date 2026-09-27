"""Controlled GQA ablation: same tokenizer, documents, updates and target budget."""

import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from modern_decoder import ModernDecoder
from language_model import sequence_loss
from tokenization import PreSplitBPE
from text_pretraining import TRAIN_TEXT, VALID_TEXT, windows, evaluate
from evaluation import paired_bootstrap


def verify():
    torch.set_num_threads(1)
    tokenizer = PreSplitBPE(["<bos>", "<eos>"]).fit(TRAIN_TEXT, 16)
    training = windows(TRAIN_TEXT, tokenizer, 24)
    validation_by_document = [windows([text], tokenizer, 24) for text in VALID_TEXT]
    records = []
    for seed in (17, 19):
        torch.manual_seed(seed)
        baseline = ModernDecoder(tokenizer.vocab_size, width=24, ff_width=48, heads=4,
                                 kv_heads=2, head_width=8, max_length=64).double()
        for kv_heads in (1, 2):
            torch.manual_seed(seed)
            model = ModernDecoder(**{**baseline.config, "kv_heads": kv_heads}).double()
            # Copy matching-shaped parameters so unchanged branches start with identical values.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    original = dict(baseline.named_parameters())[name]
                    if parameter.shape == original.shape:
                        parameter.copy_(original)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.01)
            seen = 0
            for step in range(24):
                sample = training[step % len(training)]
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(model(sample[:, :-1]), sample[:, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                seen += sample.shape[1] - 1
            scores = [evaluate(model, samples)[0] for samples in validation_by_document]
            records.append(dict(seed=seed, kv_heads=kv_heads, parameters=model.parameter_count(),
                                updates=24, training_targets=seen,
                                document_nll=scores, tokenizer=tokenizer.state_dict()))
    for first, second in zip(records[::2], records[1::2]):
        assert first["training_targets"] == second["training_targets"]
        assert first["parameters"] < second["parameters"]
        assert first["tokenizer"] == second["tokenizer"]
        first["paired_difference_2_minus_1"] = paired_bootstrap(first["document_nll"], second["document_nll"])
    print(json.dumps(dict(protocol="same target budget, not same FLOPs; two validation documents are not sufficient for significance",
                         records=records), indent=2))
    print("PASS: controlled KV-head ablation, matching branch initialization, two seeds and document-paired scores")


if __name__ == "__main__":
    verify()
