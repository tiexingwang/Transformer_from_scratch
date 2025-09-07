import pytest
from src.utils.vocabulary_builder import VocabularyBuilder
import torch
from torch.testing import assert_close

def test_vocabulary_builder():
    vocab_builder = VocabularyBuilder(max_vocab_size=100)
    # define some syt
    english_texts = ["hello", "world"]
    chinese_texts = ["你", "好", "世", "界"]
    # build vocabulary
    
    en_word2idx, zh_char2idx = vocab_builder.build_vocabulary(english_texts, chinese_texts)
    print(en_word2idx)
    print(zh_char2idx)
    # test vocabulary size need to think about the special tokens
    assert len(en_word2idx) == (2+4)
    assert len(zh_char2idx) == (4+4)
    # test vocabulary content
    assert en_word2idx["<pad>"] == 0
    assert en_word2idx["<unk>"] == 1
    assert en_word2idx["<bos>"] == 2
    assert en_word2idx["<eos>"] == 3
    assert zh_char2idx["<pad>"] == 0
    assert zh_char2idx["<unk>"] == 1
    assert zh_char2idx["<bos>"] == 2
    assert zh_char2idx["<eos>"] == 3
    assert en_word2idx["hello"] == 4
    assert en_word2idx["world"] == 5
    assert zh_char2idx["你"] == 4
    assert zh_char2idx["好"] == 5
    assert zh_char2idx["世"] == 6
    assert zh_char2idx["界"] == 7

if __name__ == "__main__":
    pytest.main()

