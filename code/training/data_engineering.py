"""HTML filtering, audited near-duplicate grouping and source-mixture checks."""

from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import math
import random
import re


class TextExtractor(HTMLParser):
    """Fixed teaching HTML rules; not a general main-content detector."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.ignored = [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.ignored.append(tag)
        if not self.ignored and tag in {"p", "div", "br", "li", "h1", "h2"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if self.ignored and tag == self.ignored[-1]:
            self.ignored.pop()
        if not self.ignored and tag in {"p", "div", "li", "h1", "h2", "article", "script", "style", "nav", "header", "footer"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def extract_text(html):
    parser = TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join("".join(parser.parts).split())


def filter_record(text, min_words=6, min_letter_fraction=.5):
    if min_words < 1 or not 0 <= min_letter_fraction <= 1:
        raise ValueError("invalid filtering thresholds")
    if len(text.split()) < min_words:
        return "too_short"
    visible = [c for c in text if not c.isspace()]
    if sum(c.isalpha() for c in visible) / max(1, len(visible)) < min_letter_fraction:
        return "low_letter_fraction"
    if re.search(r"\b(?:password|api_key)\s*[:=]\s*\S+", text, re.IGNORECASE):
        return "possible_secret"
    return "keep"


def shingles(text, width=3):
    if width < 1:
        raise ValueError("shingle width must be positive")
    tokens = re.findall(r"\w+", text.lower())
    return {" ".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.


def minhash(items, count=120):
    if not items or count < 1:
        raise ValueError("MinHash needs a nonempty set and positive signature length")
    return tuple(min(int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8,
                                                  key=seed.to_bytes(8, "little")).digest(), "big")
                     for item in items) for seed in range(count))


def lsh_candidates(signatures, bands):
    if not signatures or bands < 1 or len(signatures[0]) % bands:
        raise ValueError("signature length must be divisible by positive bands")
    rows = len(signatures[0]) // bands
    if not rows or any(len(s) != bands * rows for s in signatures):
        raise ValueError("signature lengths must agree and be nonempty")
    buckets, candidates = defaultdict(list), set()
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = band, signature[band * rows:(band + 1) * rows]
            candidates.update((previous, index) for previous in buckets[key])
            buckets[key].append(index)
    return candidates


def duplicate_groups(texts, threshold=.6, use_lsh=True):
    if not 0 < threshold <= 1 or not texts:
        raise ValueError("need documents and 0 < threshold <= 1")
    sets = [shingles(text) for text in texts]
    if any(not s for s in sets):
        raise ValueError("filter short documents before near-duplicate grouping")
    if use_lsh:
        candidates = lsh_candidates([minhash(s) for s in sets], 60)
    else:
        # ponytail: quadratic exact baseline only for small teaching corpora; LSH generates scalable candidates.
        candidates = {(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))}
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in sorted(candidates):
        if jaccard(sets[i], sets[j]) >= threshold:
            parent[root(j)] = root(i)
    groups = defaultdict(list)
    for index in range(len(texts)):
        groups[root(index)].append(index)
    return list(groups.values()), candidates


def split_groups(groups, seed=5):
    if len(groups) < 2:
        raise ValueError("need at least two independent groups for train/validation")
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    validation = {i for i in groups[order[0]]}
    training = {i for group in order[1:] for i in groups[group]}
    return training, validation


def mixture_report(probabilities, source_tokens, training_tokens, epoch_cap):
    if set(probabilities) != set(source_tokens) or not probabilities:
        raise ValueError("mixture and token-count sources must match")
    if not math.isfinite(training_tokens) or training_tokens <= 0 or not math.isfinite(epoch_cap) or epoch_cap <= 0:
        raise ValueError("training tokens and epoch cap must be positive")
    if any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1.):
        raise ValueError("source probabilities must be finite, nonnegative and sum to one")
    if any(not math.isfinite(n) or n <= 0 for n in source_tokens.values()):
        raise ValueError("source token counts must be positive")
    epochs = {source: p * training_tokens / source_tokens[source] for source, p in probabilities.items()}
    return dict(epochs=epochs, over_cap=[s for s, e in epochs.items() if e > epoch_cap])


def verify():
    html = "<nav>links</nav><article><p>the cat watches the quiet garden in the rain.</p><script>secret()</script></article>"
    text = extract_text(html)
    assert text == "the cat watches the quiet garden in the rain."
    assert extract_text("<p>a &amp; b<br>c</p>") == "a & b c"
    assert extract_text("<p>he<em>ll</em>o</p><p>world</p>") == "hello world"
    assert filter_record(text) == "keep"
    assert filter_record("buy now") == "too_short"
    assert filter_record("123 456 789 012 345 678") == "low_letter_fraction"
    assert filter_record("please do not share this password=abc with anyone") == "possible_secret"
    documents = [text, text.replace("quiet", "green"), text,
                 "a dog sleeps beside the door while a child reads a book.",
                 "the sun warms the street and a bird sings in the tree."]
    groups, candidates = duplicate_groups(documents, threshold=.4)
    exact_groups, _ = duplicate_groups(documents, threshold=.4, use_lsh=False)
    assert groups == exact_groups == [[0, 1, 2], [3], [4]]
    assert (0, 2) in candidates
    train, valid = split_groups(groups)
    assert train.isdisjoint(valid) and train | valid == set(range(len(documents)))
    assert all(set(g) <= train or set(g) <= valid for g in groups)
    a, b = shingles(documents[0]), shingles(documents[1])
    signature_a, signature_b = minhash(a, 1000), minhash(b, 1000)
    estimate = sum(x == y for x, y in zip(signature_a, signature_b)) / 1000
    assert abs(estimate - jaccard(a, b)) < .08
    report = mixture_report({"web": .5, "curated": .5}, {"web": 10000, "curated": 100}, 10000, 3)
    assert report["epochs"] == {"web": .5, "curated": 50.} and report["over_cap"] == ["curated"]
    draws = random.Random(5).choices(["web", "curated"], weights=[.8, .2], k=10000)
    assert abs(draws.count("curated") / len(draws) - .2) < .02
    print(f"PASS: HTML rules/filter reasons; exact-confirmed LSH groups={groups}; Jaccard={jaccard(a,b):.3f}, MinHash={estimate:.3f}")
    print("PASS: duplicate-group split; seeded source resampling; 50-epoch scarce-source warning")


if __name__ == "__main__":
    verify()
