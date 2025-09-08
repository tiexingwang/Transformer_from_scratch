import torch

class VocabularyBuilder:
    def __init__(self, max_vocab_size=1000):
        """
        Initialize the VocabularyBuilder
        Args:
            max_vocab_size: The maximum size of the vocabulary
        Returns:
            None
        """
        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.BOS_TOKEN = "<bos>"
        self.EOS_TOKEN = "<eos>"
        self.max_vocab_size = max_vocab_size
        
    def read_data_from_file(self, file_path):
        """Read data from file and extract text all the text
        Args:
            file_path: The path to the file
        Returns:
            english_texts: The list of English texts
            chinese_texts: The list of Chinese texts
        """
        english_texts = []
        chinese_texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                english_text, chinese_text = line.split('\t')
                english_texts.append(english_text)
                chinese_texts.append(chinese_text)
        return english_texts, chinese_texts
    
    def build_english_vocabulary(self, english_texts):
        """Build English vocabulary library from data
        Args:
            english_texts: The list of English texts
        Returns:
            word2idx: The dictionary of word to index
            for example:
                {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<bos>": 2,
                    "<eos>": 3,
                    "hello": 4,
                    "world": 5
                }
        """
        word_count = {}
        
        # Count word frequency
        for text in english_texts:
            words = text.lower().split() # Convert to lowercase to avoid case sensitivity
            for word in words:
                if word not in word_count:
                    word_count[word] = 0
                word_count[word] += 1

        # Start with special tokens
        vocabulary = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        
        # Sort words by frequency and add most frequent ones
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if len(vocabulary) < self.max_vocab_size:
                vocabulary.append(word)
            else:
                break
                
        # Create word-to-index mapping
        word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        return word2idx

    def build_chinese_vocabulary(self, chinese_texts):
        """Build Chinese vocabulary library from data
        Args:
            chinese_texts: The list of Chinese texts
        Returns:
            char2idx: The dictionary of character to index
            for example:
                {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<bos>": 2,
                    "<eos>": 3,
                    "你": 4,
                    "好": 5,
                    "世": 6,
                    "界": 7
                }
        """
        char_count = {}
        for text in chinese_texts:
            for char in text:
                if char not in char_count:
                    char_count[char] = 0
                char_count[char] += 1

        # Start with special tokens
        vocabulary = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        # Sort characters by frequency and add most frequent ones
        sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars:
            if len(vocabulary) < self.max_vocab_size:
                vocabulary.append(char)
            else:
                break
        # Create character-to-index mapping
        char2idx = {char: idx for idx, char in enumerate(vocabulary)}
        return char2idx
    
    def build_vocabulary(self, english_texts, chinese_texts):
        """Build vocabulary from data
        Args:
            english_texts: The list of English texts
            chinese_texts: The list of Chinese texts
        Returns:
            en_word2idx: The dictionary of word to index
            zh_char2idx: The dictionary of character to index
        """
        en_word2idx = self.build_english_vocabulary(english_texts)
        zh_char2idx = self.build_chinese_vocabulary(chinese_texts)
        return en_word2idx, zh_char2idx

if __name__ == "__main__":
    vocab_builder = VocabularyBuilder(max_vocab_size=100)
    english_texts, chinese_texts = vocab_builder.read_data_from_file("data/sample.en-zh.txt")
    en_word2idx, zh_char2idx = vocab_builder.build_vocabulary(english_texts, chinese_texts)
    print("English vocabulary (word to index):")
    print(en_word2idx)
    print("\nChinese vocabulary (character to index):")
    print(zh_char2idx)

    # English vocabulary (word to index):
    # {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, 'you': 4, 'hello': 5, 'world': 6, 'how': 7, 'are': 8, 'i': 9, 'love': 10, 'good': 11, 'morning': 12, 'thank': 13}    #
    # Chinese vocabulary (character to index):
    # {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '\n': 4, '你': 5, '好': 6, '谢': 7, '世': 8, '界': 9, '吗': 10, '我': 11, '爱': 12, '早': 13, '上': 14}

