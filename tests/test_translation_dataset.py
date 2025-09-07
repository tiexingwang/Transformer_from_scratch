import pytest
import torch
import os
import tempfile
from torch.utils.data import DataLoader
from src.utils.translation_dataset import TranslationDataset
from src.utils.vocabulary_builder import VocabularyBuilder

@pytest.fixture
def vocab_builder():
    return VocabularyBuilder(max_vocab_size=10000)

def test_translation_dataset_basic(vocab_builder):
    """Test basic functionality of TranslationDataset"""
    
    # Create temporary file with sample data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello world\t你好世界\n")
        f.write("Good morning\t早上好\n")
        f.write("How are you\t你好吗\n")
        temp_file = f.name
    
    try:
        # Create dataset
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=10)
        
        # Test dataset length
        assert len(dataset) == 3
        
        # Test getting an item
        source, target = dataset[0]
        
        # Test tensor properties
        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert source.dtype == torch.long
        assert target.dtype == torch.long
        assert source.shape == (10,)
        assert target.shape == (10,)
        
        # Test that all items have same shape
        for i in range(len(dataset)):
            src, tgt = dataset[i]
            assert src.shape == (10,)
            assert tgt.shape == (10,)
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

def test_translation_dataset_different_lengths(vocab_builder):
    """Test dataset with different max_length values"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello\t你好\n")
        f.write("World\t世界\n")
        temp_file = f.name
    
    try:
        # Test with different max_length values
        for max_len in [5, 10, 20]:
            dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=max_len)
            source, target = dataset[0]
            assert source.shape == (max_len,)
            assert target.shape == (max_len,)
        
    finally:
        os.unlink(temp_file)

def test_translation_dataset_file_not_found(vocab_builder):
    """Test error handling for missing files"""
    with pytest.raises(FileNotFoundError):
        TranslationDataset(source_file="nonexistent_file.txt", vocabulary_builder=vocab_builder, max_length=10)

def test_translation_dataset_with_dataloader(vocab_builder):
    """Test dataset compatibility with PyTorch DataLoader"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello world\t你好世界\n")
        f.write("Good morning\t早上好\n")
        f.write("How are you\t你好吗\n")
        f.write("I love you\t我爱你\n")
        temp_file = f.name
    
    try:
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=8)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Test that DataLoader works
        batches = list(dataloader)
        assert len(batches) == 2  # 4 samples / 2 batch_size = 2 batches
        
        # Test first batch
        source_batch, target_batch = batches[0]
        assert source_batch.shape == (2, 8)  # batch_size=2, max_length=8
        assert target_batch.shape == (2, 8)
        
        # Test second batch
        source_batch, target_batch = batches[1]
        assert source_batch.shape == (2, 8)
        assert target_batch.shape == (2, 8)
        
    finally:
        os.unlink(temp_file)

def test_translation_dataset_token_consistency(vocab_builder):
    """Test that tokenization is consistent across multiple calls"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello world\t你好世界\n")
        temp_file = f.name
    
    try:
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=10)
        
        # Get the same item multiple times
        source1, target1 = dataset[0]
        source2, target2 = dataset[0]
        
        # Should be identical
        assert torch.equal(source1, source2)
        assert torch.equal(target1, target2)
        
    finally:
        os.unlink(temp_file)

def test_translation_dataset_vocabulary_properties(vocab_builder):
    """Test vocabulary-related properties"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hello world\t你好世界\n")
        f.write("Good morning\t早上好\n")
        temp_file = f.name
    
    try:
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=10)
        
        # Test that vocabularies are created
        assert hasattr(dataset, 'vocab_en')
        assert hasattr(dataset, 'vocab_zh')
        assert isinstance(dataset.vocab_en, dict)
        assert isinstance(dataset.vocab_zh, dict)
        
        # Test that special tokens exist
        assert '<unk>' in dataset.vocab_en
        assert '<unk>' in dataset.vocab_zh
        assert '<bos>' in dataset.vocab_en
        assert '<bos>' in dataset.vocab_zh
        assert '<eos>' in dataset.vocab_en
        assert '<eos>' in dataset.vocab_zh
        assert '<pad>' in dataset.vocab_en
        assert '<pad>' in dataset.vocab_zh
        
    finally:
        os.unlink(temp_file)

def test_translation_dataset_edge_cases(vocab_builder):
    """Test edge cases and boundary conditions"""
    
    # Test with very short max_length
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("Hi\t你好\n")
        temp_file = f.name
    
    try:
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=3)
        source, target = dataset[0]
        assert source.shape == (3,)
        assert target.shape == (3,)
        
    finally:
        os.unlink(temp_file)

def test_translation_dataset_large_file(vocab_builder):
    """Test with a larger dataset"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        # Create 10 sample pairs
        for i in range(10):
            f.write(f"Hello {i}\t你好{i}\n")
        temp_file = f.name
    
    try:
        dataset = TranslationDataset(source_file=temp_file, vocabulary_builder=vocab_builder, max_length=5)
        assert len(dataset) == 10
        
        # Test DataLoader with larger dataset
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        batches = list(dataloader)
        assert len(batches) == 4  # 10 samples / 3 batch_size = 4 batches (3+3+3+1)
        
    finally:
        os.unlink(temp_file)