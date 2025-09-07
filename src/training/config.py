from dataclasses import dataclass
from pathlib import Path
import argparse

@dataclass
class TrainerConfig:
    # === Data parameters ===
    train_dataset_path: Path  # Directory path where the dataset is located
    val_dataset_path: Path  # Directory path where the dataset is located
    max_length: int = 32  # Maximum sequence length for tokenization/padding
    max_vocab_size: int = 10000  # Maximum vocabulary size for both source and target languages

    # === Model parameters ===
    d_model: int = 64  # Embedding dimension (model hidden size)
    n_heads: int = 4  # Number of attention heads in the Transformer
    n_layers: int = 2  # Number of encoder/decoder layers in the Transformer
    d_ff: int = 256  # Feed-forward network hidden dimension
    dropout: float = 0.1  # Dropout rate for regularization

    # === Training parameters ===
    lr: float = 0.001  # Learning rate for the optimizer
    step_size: int = 10  # Step size for learning rate scheduler
    gamma: float = 0.1  # Multiplicative factor of learning rate decay
    batch_size: int = 32  # Number of samples per batch
    shuffle: bool = True  # Whether to shuffle the dataset during training
    num_epochs: int = 100  # Number of training epochs
    gpu_id: int = 0  # GPU device ID to use for training (if available)

    # === Logging and checkpoint parameters ===
    log_interval: int = 10  # How often (in steps) to log training progress
    save_interval: int = 10  # How often (in epochs) to save the model
    best_model_path: Path = Path("best_model.pt")  # Path to save the best model checkpoint
    best_model_name: str = "best_model.pt"  # File name for the best model checkpoint

    def validate(self):
        # Accept both str and Path for dataset paths, and check existence accordingly
        for p, label in [(self.train_dataset_path, "train_dataset_path"), (self.val_dataset_path, "val_dataset_path")]:
            path_obj = Path(p) if not isinstance(p, Path) else p
            if not path_obj.exists():
                raise ValueError(f"Dataset path {label} ({p}) does not exist")
        if self.max_length <= 0:
            raise ValueError("Max length must be greater than 0")
        if self.max_vocab_size <= 0:
            raise ValueError("Max vocabulary size must be greater than 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be greater than 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be greater than 0")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be greater than 0")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")
        if self.lr <= 0:
            raise ValueError("Learning rate must be greater than 0")
        if self.step_size <= 0:
            raise ValueError("Step size must be greater than 0")
        if self.gamma <= 0:
            raise ValueError("Gamma must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0")
        if self.log_interval <= 0:
            raise ValueError("Log interval must be greater than 0")
        if self.save_interval <= 0:
            raise ValueError("Save interval must be greater than 0")
        # No need to check if best_model_path exists; it's fine if it already exists.
        if self.best_model_name==None:
            raise ValueError("Best model name is required")
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    