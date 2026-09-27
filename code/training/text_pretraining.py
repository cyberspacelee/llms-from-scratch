"""Modern LM on original short prose: train, validate, save, resume, generate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from language_model import sequence_loss
from modern_decoder import ModernDecoder, generate_cached
from tokenization import ByteBPE, PreSplitBPE


# Original teaching prose; each string is one document, never packed across boundaries.
TRAIN_TEXT = [
    "the cat sits by the window. the cat watches the rain.",
    "the dog waits by the door. the dog watches the street.",
    "a bird sings in the garden. a bird rests in the tree.",
    "the rain falls on the garden. the rain stops at night.",
    "a child reads a small book. a child closes the book.",
    "the sun warms the street. the sun lights the window.",
]
VALID_TEXT = [
    "the cat rests by the door. the cat watches the garden.",
    "a child watches the rain. a bird sits in the tree.",
]


def windows(documents, tokenizer, context):
    samples = []
    for text in documents:
        ids = [tokenizer.special_ids["<bos>"]] + tokenizer.encode(text) + [tokenizer.special_ids["<eos>"]]
        for start in range(0, len(ids) - 1, context):
            samples.append(torch.tensor([ids[start:start + context + 1]], dtype=torch.long))
    if not samples:
        raise ValueError("documents must yield at least one prediction target")
    return samples


def data_fingerprint(documents, tokenizer):
    payload = json.dumps(dict(documents=documents, tokenizer=tokenizer.state_dict()),
                         sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_run(train_documents, validation_documents, seed=29):
    if not train_documents or not validation_documents or any(not text.strip() for text in train_documents + validation_documents):
        raise ValueError("training and validation need nonempty documents")
    if set(train_documents) & set(validation_documents):
        raise ValueError("training and validation documents must not overlap")
    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = PreSplitBPE(["<bos>", "<eos>"]).fit(train_documents, 16)
    model = ModernDecoder(tokenizer.vocab_size, width=24, ff_width=48, layers=2,
                          heads=4, kv_heads=2, head_width=8, max_length=64).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.01, foreach=False)
    samples = windows(train_documents, tokenizer, 24)
    state = dict(step=0, cursor=0, order=list(range(len(samples))), context=24, seed=seed)
    return model, optimizer, tokenizer, samples, state


def train_steps(model, optimizer, samples, state, count):
    if type(count) is not int or count < 0:
        raise ValueError("step count must be a nonnegative integer")
    model.train()
    losses = []
    for _ in range(count):
        if state["cursor"] == len(state["order"]):
            state["order"] = torch.randperm(len(samples)).tolist()
            state["cursor"] = 0
        document = samples[state["order"][state["cursor"]]]
        state["cursor"] += 1
        # Random suffix augmentation consumes the Python RNG; shuffle consumes the Torch RNG.
        document = document[:, random.randrange(min(3, document.shape[1] - 1)):]
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model(document[:, :-1]), document[:, 1:])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        state["step"] += 1
        state["last_grad_norm"] = float(norm)
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, samples):
    model.eval()
    report = []
    for index, document in enumerate(samples):
        loss = sequence_loss(model(document[:, :-1]), document[:, 1:]).item()
        report.append(dict(sample=index, targets=document.shape[1] - 1, nll=loss))
    targets = sum(row["targets"] for row in report)
    return sum(row["nll"] * row["targets"] for row in report) / targets, report


def save_checkpoint(path, model, optimizer, tokenizer, state, documents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, config=model.config, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), tokenizer=tokenizer.state_dict(),
                   training=state, documents=documents,
                   data_hash=data_fingerprint(documents, tokenizer),
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    # Write on the destination filesystem, then atomically replace the previous checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    tokenizer = ByteBPE.from_state_dict(payload["tokenizer"])
    if tokenizer.vocab_size != payload["config"]["vocab_size"]:
        raise ValueError("checkpoint model and tokenizer vocabulary disagree")
    if data_fingerprint(payload["documents"], tokenizer) != payload["data_hash"]:
        raise ValueError("checkpoint data or tokenizer fingerprint mismatch")
    model = ModernDecoder(**payload["config"]).double()
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    optimizer.load_state_dict(payload["optimizer"])
    state = payload["training"]
    samples = windows(payload["documents"]["train"], tokenizer, state["context"])
    if sorted(state["order"]) != list(range(len(samples))) or not 0 <= state["cursor"] <= len(samples):
        raise ValueError("checkpoint data cursor/order is invalid")
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    return model, optimizer, tokenizer, samples, state, payload["documents"]


def verify():
    torch.set_num_threads(1)
    model, optimizer, tokenizer, samples, state = new_run(TRAIN_TEXT, VALID_TEXT)
    validation = windows(VALID_TEXT, tokenizer, state["context"])
    initial, _ = evaluate(model, validation)
    train_steps(model, optimizer, samples, state, 20)
    documents = dict(train=TRAIN_TEXT, validation=VALID_TEXT)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        save_checkpoint(checkpoint, model, optimizer, tokenizer, state, documents)
        reference_logits = model(samples[0][:, :-1]).detach()
        losses = train_steps(model, optimizer, samples, state, 8)
        expected = copy.deepcopy(model.state_dict())
        expected_state = copy.deepcopy(state)
        restored, restored_optimizer, loaded_tokenizer, loaded_samples, loaded_state, _ = load_checkpoint(checkpoint)
        torch.testing.assert_close(restored(loaded_samples[0][:, :-1]), reference_logits, rtol=0, atol=0)
        assert loaded_tokenizer.state_dict() == tokenizer.state_dict()
        assert train_steps(restored, restored_optimizer, loaded_samples, loaded_state, 8) == losses
        assert loaded_state == expected_state
        for name, parameter in restored.state_dict().items():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        final, report = evaluate(restored, validation)
        assert final < initial
        assert sum(row["targets"] for row in report) == sum(s.shape[1] - 1 for s in validation)
        prompt = torch.tensor([[tokenizer.special_ids["<bos>"]] + tokenizer.encode("the cat")])
        output = generate_cached(restored, prompt, 8, tokenizer.special_ids["<eos>"],
                                 torch.Generator().manual_seed(4))
        assert torch.equal(output[:, :prompt.shape[1]], prompt)
    print(f"PASS: prose training validation NLL {initial:.6f} -> {final:.6f}; targets={sum(r['targets'] for r in report)}")
    print("PASS: disk checkpoint/tokenizer/logits; exact next eight updates, cursor and RNG recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80, help="additional updates on this invocation")
    parser.add_argument("--train-text", type=Path, help="UTF-8 file: one document per nonempty line")
    parser.add_argument("--validation-text", type=Path)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--prompt", default="the cat")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 0 or args.max_new_tokens < 0:
        parser.error("steps and max-new-tokens must be nonnegative")
    if bool(args.train_text) != bool(args.validation_text):
        parser.error("provide both training and validation files")
    if args.checkpoint is None:
        if args.resume or args.generate_only or args.train_text:
            parser.error("these options require --checkpoint")
        verify()
        return
    torch.set_num_threads(1)
    if args.resume or args.generate_only:
        model, optimizer, tokenizer, samples, state, documents = load_checkpoint(args.checkpoint)
        if args.train_text:
            parser.error("resume uses the checkpoint's frozen documents and tokenizer")
    else:
        if args.checkpoint.exists():
            parser.error("checkpoint already exists; use --resume, --generate-only, or a new path")
        documents = dict(train=TRAIN_TEXT, validation=VALID_TEXT)
        if args.train_text:
            documents = {key: [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                         for key, path in (("train", args.train_text), ("validation", args.validation_text))}
        model, optimizer, tokenizer, samples, state = new_run(documents["train"], documents["validation"], args.seed)
    validation = windows(documents["validation"], tokenizer, state["context"])
    if not args.generate_only:
        train_steps(model, optimizer, samples, state, args.steps)
        save_checkpoint(args.checkpoint, model, optimizer, tokenizer, state, documents)
        tokenizer.save(args.checkpoint.with_suffix(".tokenizer.json"))
    nll, report = evaluate(model, validation)
    print(json.dumps(dict(step=state["step"], validation_nll=nll, samples=report), indent=2))
    prompt = torch.tensor([[tokenizer.special_ids["<bos>"]] + tokenizer.encode(args.prompt)])
    output = generate_cached(model, prompt, args.max_new_tokens, tokenizer.special_ids["<eos>"],
                             torch.Generator().manual_seed(args.seed))
    print(json.dumps(dict(generated_text=tokenizer.decode(output[0].tolist(), errors="replace")), ensure_ascii=False))


if __name__ == "__main__":
    main()
