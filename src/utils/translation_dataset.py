import torch
from torch.utils.data import Dataset

# Handle imports for both relative and absolute paths
try:
    from .vocabulary_builder import VocabularyBuilder
    from .tokenizer import Tokenizer
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.utils.vocabulary_builder import VocabularyBuilder
    from src.utils.tokenizer import Tokenizer

class TranslationDataset(Dataset):
    def __init__(self, source_file, vocabulary_builder, max_length=32):
        self.source_file = source_file

        self.max_length = max_length

        # build vocabulary
        self.vocab_builder = vocabulary_builder

        # read data from file 
        self.english_texts, self.chinese_texts = self.vocab_builder.read_data_from_file(self.source_file)
        self.vocab_en, self.vocab_zh = self.vocab_builder.build_vocabulary(self.english_texts, self.chinese_texts)

        # build tokenizer
        self.tokenizer_en = Tokenizer(self.vocab_en, max_length=self.max_length)
        self.tokenizer_zh = Tokenizer(self.vocab_zh, max_length=self.max_length)

        # tokenize the data
        self.en_tokens = self.tokenizer_en.tokenize_and_pad_and_truncate_batch(self.english_texts)
        self.zh_tokens = self.tokenizer_zh.tokenize_and_pad_and_truncate_batch(self.chinese_texts)

    def __len__(self):
        return len(self.en_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.en_tokens[idx], dtype=torch.long), torch.tensor(self.zh_tokens[idx], dtype=torch.long)


if __name__ == "__main__":
    # Example usage of TranslationDataset and DataLoader

    # 1. Build a vocabulary from your data file
    vocab_builder = VocabularyBuilder(max_vocab_size=10000)

    # 2. Create a TranslationDataset instance
    #    - source_file: path to your parallel corpus (tab-separated English and Chinese)
    #    - vocabulary_builder: an instance of VocabularyBuilder
    #    - max_length: maximum sequence length for tokenization/padding
    dataset = TranslationDataset(
        source_file="data/sample.en-zh.txt", 
        vocabulary_builder=vocab_builder,
        max_length=32
    )

    # 3. Access a single (source, target) tokenized pair as tensors
    print(dataset[0])  # prints (source_tensor, target_tensor) for the first example

    # 4. Get the total number of samples in the dataset
    print(len(dataset))

    # 5. Using DataLoader for batching and shuffling
    #    - Useful for training loops in PyTorch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 6. Iterate over batches

    print("-"*100)
    i = 0
    for batch in dataloader:
        source, target = batch  # source and target are batches of tokenized tensors
        print(source)
        print(target)
        print(f"{i}-"*100)
        i += 1

        if i > 10:
            break  # Remove this break to iterate over the whole dataset

