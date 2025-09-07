from .vocabulary_builder import VocabularyBuilder

# After building the vocabulary, we can use the vocabulary to tokenize the text.
class Tokenizer:
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        self.idx2word = {idx: word for word, idx in vocab.items()}
    
    def tokenize_english(self, text):
        """Tokenize English text"""

        # split the text into words
        words = text.lower().split()
        # tokenize the words
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab["<unk>"])
        return tokens
    
    def tokenize_chinese(self, text):
        """Tokenize Chinese text"""
        # split the text into characters
        chars = list(text)
        # tokenize the characters
        tokens = []
        for char in chars: 
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab["<unk>"])
        return tokens
    
    def tokenize(self, text):

        """Tokenize text"""

        if not isinstance(text, str):
            raise ValueError("Invalid text type")
        
        if self.detect_language(text) == 'chinese':
            return self.tokenize_chinese(text)
        else:
            return self.tokenize_english(text)       
    
    def detect_language(self, text):
        """Detect if text is Chinese or English"""
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return 'chinese' if chinese_chars > len(text) * 0.3 else 'english'

    def add_special_tokens(self, tokens):
        """Add special tokens to the tokens"""
        return [self.vocab["<bos>"]] + tokens + [self.vocab["<eos>"]]
    
    def pad_tokens(self, tokens, max_length):
        """Pad the tokens to the max length"""
        return tokens + [self.vocab["<pad>"]] * (max_length - len(tokens))
    
    def truncate_tokens(self, tokens, max_length):
        """Truncate the tokens to the max length"""
        return tokens[:max_length]
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        return len(self.vocab)
    
    def get_vocab(self):
        """Get the vocabulary"""
        return self.vocab
    
    def tokenize_and_pad_and_truncate(self, text):
        """Tokenize and pad the text"""
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        tokens = self.pad_tokens(tokens, self.max_length)
        tokens = self.truncate_tokens(tokens, self.max_length)
        return tokens
    
    def tokenize_and_pad_and_truncate_batch(self, texts):
        """Tokenize and pad and truncate the batch of text"""
        tokens = [self.tokenize_and_pad_and_truncate(text) for text in texts]
        return tokens
    
    def detokenize(self, tokens):
        """Detokenize the tokens, skipping special tokens like <pad>, <bos>, <eos>."""
        detokenized_text = []
        for token in tokens:
            word = self.idx2word[token] if token in self.idx2word else self.idx2word["<unk>"]
            # Skip special tokens
            if word in {"<pad>", "<bos>", "<eos>"}:
                continue
            detokenized_text.append(word)
        return " ".join(detokenized_text)
    
    def detokenize_batch(self, tokens):
        """Detokenize the batch of tokens"""
        return [self.detokenize(token) for token in tokens]

if __name__ == "__main__":
    vocab_builder = VocabularyBuilder(max_vocab_size=100)

    # create sample data
    english_texts = ["Hello world"]
    chinese_texts = ["你好世界"]

    # build vocabulary
    en_word2idx, zh_char2idx = vocab_builder.build_vocabulary(english_texts, chinese_texts)
    print(f"English vocabulary: {en_word2idx}   \nChinese vocabulary: {zh_char2idx}")

    tokenizer_en = Tokenizer(en_word2idx, max_length=20)
    tokenizer_zh = Tokenizer(zh_char2idx, max_length=20)

    # tokenize and pad and truncate the text
    print(tokenizer_en.tokenize_and_pad_and_truncate("Hello world"))
    print(tokenizer_zh.tokenize_and_pad_and_truncate("你好世界"))

    # detokenize the text
    print(tokenizer_en.detokenize(tokenizer_en.tokenize_and_pad_and_truncate("Hello world")))
    print(tokenizer_zh.detokenize(tokenizer_zh.tokenize_and_pad_and_truncate("你好世界")))

    # create batch of text
    english_texts_batch = ["Hello world", "Hello United kingdom"]
    chinese_texts_batch = ["你好世界", "你好英国"]
    
    # redefine the vocabulary in batch 
    en_word2idx_batch, zh_char2idx_batch = vocab_builder.build_vocabulary(english_texts_batch, chinese_texts_batch)
    tokenizer_en_batch = Tokenizer(en_word2idx_batch, max_length=20)
    tokenizer_zh_batch = Tokenizer(zh_char2idx_batch, max_length=20)
    # tokenize and pad and truncate the batch of text
    print(tokenizer_en_batch.tokenize_and_pad_and_truncate_batch(english_texts_batch))
    print(tokenizer_zh_batch.tokenize_and_pad_and_truncate_batch(chinese_texts_batch))
    # detokenize the batch of tokens
    print(tokenizer_en_batch.detokenize_batch(tokenizer_en_batch.tokenize_and_pad_and_truncate_batch(english_texts_batch)))
    print(tokenizer_zh_batch.detokenize_batch(tokenizer_zh_batch.tokenize_and_pad_and_truncate_batch(chinese_texts_batch)))

    