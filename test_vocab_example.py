#!/usr/bin/env python3
"""
Simple example to demonstrate vocabulary builder output
"""

import sys
import os
sys.path.append('src')

from utils.vocabulary_builder import VocabularyBuilder

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = [
        "Hello world\t你好世界",
        "How are you\t你好吗", 
        "I love you\t我爱你",
        "Good morning\t早上好",
        "Thank you\t谢谢"
    ]
    
    # Create sample file
    os.makedirs("data", exist_ok=True)
    with open("data/sample.en-zh.txt", "w", encoding="utf-8") as f:
        for line in sample_data:
            f.write(line + "\n")
    
    print("📝 Created sample data file: data/sample.en-zh.txt")
    return "data/sample.en-zh.txt"

def demonstrate_vocabulary_builder():
    """Demonstrate vocabulary builder step by step"""
    print("🚀 Vocabulary Builder Demonstration")
    print("=" * 50)
    
    # Create sample data
    file_path = create_sample_data()
    
    # Initialize vocabulary builder
    vocab_builder = VocabularyBuilder(max_vocab_size=100)
    
    # Step 1: Read data
    print("\n📖 Step 1: Reading data from file...")
    english_texts, chinese_texts = vocab_builder.read_data_from_file(file_path)
    
    print(f"English texts: {english_texts}")
    print(f"Chinese texts: {chinese_texts}")
    
    # Step 2: Build English vocabulary
    print("\n🏗️ Step 2: Building English vocabulary...")
    en_word2idx = vocab_builder.build_english_vocabulary(english_texts)
    
    print("English vocabulary (word -> index):")
    for word, idx in list(en_word2idx.items())[:10]:  # Show first 10
        print(f"  '{word}' -> {idx}")
    print(f"  ... (total {len(en_word2idx)} words)")
    
    # Step 3: Show tokenization process
    print("\n🔢 Step 3: Tokenization example...")
    sample_text = "Hello world"
    print(f"Original text: '{sample_text}'")
    
    # Manual tokenization to show the process
    words = sample_text.lower().split()
    print(f"Words after splitting: {words}")
    
    tokens = []
    for word in words:
        if word in en_word2idx:
            token = en_word2idx[word]
            print(f"  '{word}' -> {token}")
            tokens.append(token)
        else:
            token = en_word2idx[vocab_builder.UNK_TOKEN]
            print(f"  '{word}' -> {token} (UNK)")
            tokens.append(token)
    
    print(f"Final tokens: {tokens}")
    
    # Step 4: Show with padding
    print("\n📏 Step 4: Adding padding (max_length=8)...")
    max_length = 8
    
    # Add BOS and EOS tokens
    full_tokens = [en_word2idx[vocab_builder.BOS_TOKEN]] + tokens + [en_word2idx[vocab_builder.EOS_TOKEN]]
    print(f"With BOS/EOS: {full_tokens}")
    
    # Pad to max_length
    if len(full_tokens) > max_length:
        padded_tokens = full_tokens[:max_length]
        print(f"Truncated to {max_length}: {padded_tokens}")
    else:
        padding_needed = max_length - len(full_tokens)
        padded_tokens = full_tokens + [en_word2idx[vocab_builder.PAD_TOKEN]] * padding_needed
        print(f"Padded to {max_length}: {padded_tokens}")
    
    # Step 5: Show Chinese example
    print("\n🈶 Step 5: Chinese character example...")
    sample_chinese = "你好世界"
    print(f"Chinese text: '{sample_chinese}'")
    
    # Simple character-based tokenization
    chars = list(sample_chinese)
    print(f"Characters: {chars}")
    
    # Create a simple Chinese vocabulary for demo
    chinese_vocab = {
        vocab_builder.PAD_TOKEN: 0,
        vocab_builder.UNK_TOKEN: 1, 
        vocab_builder.BOS_TOKEN: 2,
        vocab_builder.EOS_TOKEN: 3,
        '你': 4,
        '好': 5,
        '世': 6,
        '界': 7
    }
    
    print("Chinese vocabulary:")
    for char, idx in chinese_vocab.items():
        print(f"  '{char}' -> {idx}")
    
    # Tokenize Chinese
    zh_tokens = [chinese_vocab[vocab_builder.BOS_TOKEN]]
    for char in chars:
        if char in chinese_vocab:
            zh_tokens.append(chinese_vocab[char])
        else:
            zh_tokens.append(chinese_vocab[vocab_builder.UNK_TOKEN])
    zh_tokens.append(chinese_vocab[vocab_builder.EOS_TOKEN])
    
    print(f"Chinese tokens: {zh_tokens}")
    
    # Step 6: Show final result
    print("\n🎯 Step 6: Final result for training...")
    print("English sentence: 'Hello world'")
    print(f"English tokens: {padded_tokens}")
    print("Chinese sentence: '你好世界'")
    print(f"Chinese tokens: {zh_tokens}")
    
    print("\n✅ This is what your Transformer will receive as input!")
    print("   - Numbers instead of text")
    print("   - Fixed length sequences")
    print("   - Special tokens for control")

if __name__ == "__main__":
    demonstrate_vocabulary_builder()
