from nlp_textclf.data.vocab import build_vocab, encode
from nlp_textclf.data.tokenizers import tokenize_en

def test_encode_fixed_len():
    texts = ["hello world", "hello there"]
    vocab = build_vocab(texts, tokenizer=tokenize_en, min_freq=1)
    ids = encode("hello unknown", vocab=vocab, tokenizer=tokenize_en, max_len=5)
    assert len(ids) == 5
    assert ids[0] == vocab.stoi.get("hello")
