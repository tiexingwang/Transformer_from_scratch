from src.utils.tokenizer import Tokenizer
from src.utils.vocabulary_builder import VocabularyBuilder
import pytest

def test_tokenizer_basic():
    vocab_builder = VocabularyBuilder(max_vocab_size=20)
    english_texts = ["Hello world"]
    chinese_texts = ["你好世界"]

    en_word2idx, zh_char2idx = vocab_builder.build_vocabulary(english_texts, chinese_texts)

    tokenizer_en = Tokenizer(en_word2idx, max_length=20)
    tokenizer_zh = Tokenizer(zh_char2idx, max_length=20)

    # English
    tokens_en = tokenizer_en.tokenize_and_pad_and_truncate("Hello world")
    assert tokens_en[0] == en_word2idx["<bos>"]
    assert tokens_en[1] == en_word2idx["hello"]
    assert tokens_en[2] == en_word2idx["world"]
    assert tokens_en[3] == en_word2idx["<eos>"]
    assert all(t == en_word2idx["<pad>"] for t in tokens_en[4:])

    detok_en = tokenizer_en.detokenize(tokens_en)
    assert detok_en == "hello world"

    # Chinese
    tokens_zh = tokenizer_zh.tokenize_and_pad_and_truncate("你好世界")
    assert tokens_zh[0] == zh_char2idx["<bos>"]
    for i, char in enumerate("你好世界"):
        assert tokens_zh[i+1] == zh_char2idx[char]
    assert tokens_zh[5] == zh_char2idx["<eos>"]
    assert all(t == zh_char2idx["<pad>"] for t in tokens_zh[6:])

    detok_zh = tokenizer_zh.detokenize(tokens_zh)
    assert detok_zh == "你 好 世 界"

def test_tokenizer_batch():
    vocab_builder = VocabularyBuilder(max_vocab_size=20)
    english_texts_batch = ["hello world", "hello uk"]
    chinese_texts_batch = ["你好世界", "你好英国"]

    en_word2idx_batch, zh_char2idx_batch = vocab_builder.build_vocabulary(english_texts_batch, chinese_texts_batch)
    tokenizer_en_batch = Tokenizer(en_word2idx_batch, max_length=20)
    tokenizer_zh_batch = Tokenizer(zh_char2idx_batch, max_length=20)

    # English batch
    batch_tokens_en = tokenizer_en_batch.tokenize_and_pad_and_truncate_batch(english_texts_batch)
    assert len(batch_tokens_en) == 2
    for tokens in batch_tokens_en:
        assert tokens[0] == en_word2idx_batch["<bos>"]
        assert tokens[3] == en_word2idx_batch["<eos>"]

    batch_detok_en = tokenizer_en_batch.detokenize_batch(batch_tokens_en)
    print(batch_detok_en)
    assert batch_detok_en[0] == "hello world"
    assert batch_detok_en[1].startswith("hello uk")

    # Chinese batch
    batch_tokens_zh = tokenizer_zh_batch.tokenize_and_pad_and_truncate_batch(chinese_texts_batch)
    assert len(batch_tokens_zh) == 2
    for tokens in batch_tokens_zh:
        assert tokens[0] == zh_char2idx_batch["<bos>"]
        assert tokens[5] == zh_char2idx_batch["<eos>"]

    batch_detok_zh = tokenizer_zh_batch.detokenize_batch(batch_tokens_zh)
    assert batch_detok_zh[0].replace(" ", "") == "你好世界"
    assert batch_detok_zh[1].replace(" ", "").startswith("你好英")