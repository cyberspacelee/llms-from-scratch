from llms_from_scratch.pretraining.filters import (
    c4_filter,
    gopher_quality_filter,
    gopher_repetition_filter,
    hashed_ngrams,
    mask_pii,
    train_fasttext,
)

GOOD = (
    "The river that runs through the valley has shaped the life of the town for centuries. "
    "Farmers depend on it to water their fields, and children learn to swim in its shallow bends. "
    "In spring the snow melts in the mountains and the water rises quickly, so the people have "
    "built stone walls along the banks. Each year the council meets to decide how the water "
    "should be shared with the villages downstream, and the debate is often long and heated."
)


def test_gopher_quality_keeps_prose_and_rejects_junk():
    assert gopher_quality_filter(GOOD) == (True, None)
    assert gopher_quality_filter("too short to keep")[1] == "word_count"
    assert gopher_quality_filter(" ".join(["#tag word"] * 40))[1] == "symbol_ratio"
    assert gopher_quality_filter(" ".join(["123 456 789 the and"] * 20))[1] == "alphabetic_words"


def test_gopher_repetition():
    assert gopher_repetition_filter(GOOD) == (True, None)
    spam = "\n".join(["Buy now and save big today"] * 10 + [GOOD])
    assert gopher_repetition_filter(spam)[1] == "duplicate_lines"


def test_c4_filter_drops_lines_and_documents():
    doc = "Menu\nHome | About\n" + GOOD + "\nPlease enable javascript to view this page."
    cleaned = c4_filter(doc)
    assert cleaned is not None and "Menu" not in cleaned and "javascript" not in cleaned
    assert c4_filter("Lorem ipsum dolor sit amet. " * 5) is None
    assert c4_filter("function f() { return 1; }\n" + GOOD) is None
    assert c4_filter("Only one sentence that is long enough.") is None


def test_mask_pii():
    text, counts = mask_pii("Mail alice@example.com or call (555) 123-4567 from 192.168.0.1.")
    assert counts == {"email": 1, "phone": 1, "ip": 1}
    assert "alice" not in text and "|||PHONE_NUMBER|||" in text and "|||IP_ADDRESS|||" in text


def test_hashed_ngrams_are_deterministic():
    assert hashed_ngrams("abc") == hashed_ngrams("ABC")
    assert len(hashed_ngrams("ab", 1, 2)) == 4 + 3   # "<ab>" 有 4 个 1-gram、3 个 2-gram


EN = ["the cat is on the table", "where is the train station", "i would like a cup of tea",
      "this book is very interesting", "we are going to the beach", "the weather is nice today",
      "my brother works in a bank", "please close the window"]
DE = ["die katze ist auf dem tisch", "wo ist der bahnhof", "ich möchte eine tasse tee",
      "dieses buch ist sehr interessant", "wir gehen zum strand", "das wetter ist heute schön",
      "mein bruder arbeitet in einer bank", "bitte schließen sie das fenster"]


def test_fasttext_language_id():
    texts = EN[:6] + DE[:6]
    labels = [0] * 6 + [1] * 6
    result = train_fasttext(texts, labels, num_classes=2, epochs=60)
    assert result.losses[-1] < result.losses[0] * 0.5
    probs = result.model.predict_proba(EN[6:] + DE[6:])
    assert probs.argmax(-1).tolist() == [0, 0, 1, 1]
