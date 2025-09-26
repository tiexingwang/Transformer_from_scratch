import os, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import trange
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.translation_dataset import TranslationDataset
from src.models.transformer import Transformer
from src.utils.vocabulary_builder import VocabularyBuilder
from src.utils.tokenizer import Tokenizer
from src.training.config import TrainerConfig

class Trainer:
    # The following is the class-level docstring for the Trainer class.
    # It describes the purpose of the Trainer and its expected configuration.
    #
    # Args:
    #     config (dict): Training configuration.
    #         Required keys:
    #             - train_dataset_path (str): Path to training data.
    #             - val_dataset_path (str): Path to validation data.g
    #             - max_length (int): Max tokens per sample.
    #             - d_model (int): Model hidden size (must be divisible by n_heads).
    #             - n_heads (int): Number of attention heads.
    #             - batch_size (int): Batch size.
    #             - num_epochs (int): Number of epochs.
    #         Optional keys:
    #             - dropout (float): Dropout rate. Default: 0.1
    #             - lr (float): Initial learning rate. Default: 1e-3
    #             - step_size (int): LR scheduler step. Default: 10
    #             - gamma (float): LR decay factor. Default: 0.1
    # Returns:
    #     None
    #
    # Example:
    #     config = {
    #         "train_dataset_path": "data/train.en-zh.txt",
    #         "val_dataset_path": "data/val.en-zh.txt",
    #         "max_length": 32,
    #         "d_model": 512,
    #         "n_heads": 8,
    #         "batch_size": 32,
    #         "num_epochs": 10,
    #     }
    #     trainer = Trainer(config)
    def __init__(self, config:dict):
        # === Config parameters ===
        self.config = self._build_config(config)

        # === Data parameters ===
        self.train_dataset_path = self.config.train_dataset_path
        self.val_dataset_path = self.config.val_dataset_path
        self.max_length = self.config.max_length
        self.max_vocab_size = self.config.max_vocab_size
        self.batch_size = self.config.batch_size
        self.shuffle = self.config.shuffle

        # === Model parameters ===
        self.d_model = self.config.d_model
        self.n_heads = self.config.n_heads
        self.n_layers = self.config.n_layers
        self.d_ff = self.config.d_ff
        self.dropout = self.config.dropout

        # === Training parameters ===
        self.lr = self.config.lr
        self.step_size = self.config.step_size
        self.gamma = self.config.gamma
        self.num_epochs = self.config.num_epochs
        self.gpu_id = self.config.gpu_id

        # === Logging and checkpoint parameters ===
        self.log_interval = self.config.log_interval
        self.save_interval = self.config.save_interval
        self.best_model_path = self.config.best_model_path
        self.best_model_name = self.config.best_model_name

        # === Runtime state ===
        self.device = self._get_device()
        self.vocabulary_builder = self._build_vocabulary(self.max_vocab_size)

        # === Dataset Construction ===
        self.train_dataset = self._build_train_dataset(self.train_dataset_path, self.max_length, self.vocabulary_builder)
        self.val_dataset = self._build_val_dataset(self.val_dataset_path, self.max_length, self.vocabulary_builder)

        # === Pad ID Construction ===
        self.pad_id = self.train_dataset.vocab_zh["<pad>"]

        # === Dataloader Construction ===
        self.train_dataloader = self._build_train_dataloader(self.train_dataset, self.batch_size, self.shuffle)
        self.val_dataloader = self._build_val_dataloader(self.val_dataset, getattr(self.config, "val_batch_size", self.batch_size), False)

        # === Best Loss Construction ===
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.global_step = 0

        # === Model Construction ===
        self.model = self._build_model(self.train_dataset)
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        self.scheduler = self._build_scheduler()

    # === Config and Device Utilities ===
    def _build_config(self, config):
        # This method builds and validates the configuration object.
        # It accepts either a dictionary or an argparse.Namespace as input.
        # The purpose is to allow configuration from the command line or a user-defined config file.
        # It returns a TrainerConfig object, which standardizes the configuration for the Trainer.
        if hasattr(config, "__dict__"):
            return TrainerConfig(**vars(config))
        return TrainerConfig(**config)

    def _get_device(self):
        """Select the device (GPU or CPU) for training."""
        gpu_id = self.config.gpu_id
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")

    # === Vocabulary and Dataset Construction ===
    def _build_vocabulary(self, max_vocab_size=10000):
        """Construct the vocabulary builder.
        in our case, we use the whole context of the dataset to build the vocabulary """
        return VocabularyBuilder(max_vocab_size=max_vocab_size)

    def _build_train_dataset(self, dataset_path, max_length=32, vocabulary_builder=None):
        """Create the training dataset."""
        if vocabulary_builder is None:
            raise ValueError("vocabulary_builder is required")
        if dataset_path is None:
            raise ValueError("dataset_path is required")
        return TranslationDataset(
            source_file=dataset_path,
            vocabulary_builder=vocabulary_builder,
            max_length=max_length
        )

    def _build_val_dataset(self, dataset_path, max_length=32, vocabulary_builder=None):
        """Create the validation dataset."""
        if vocabulary_builder is None:
            raise ValueError("vocabulary_builder is required")
        return TranslationDataset(
            source_file=dataset_path,
            vocabulary_builder=vocabulary_builder,
            max_length=max_length
        )

    # === DataLoader Construction ===
    def _build_train_dataloader(self, dataset, batch_size=32, shuffle=True):
        """Create the DataLoader for training."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.config, "num_workers", 0),
            pin_memory=True,
        )

    def _build_val_dataloader(self, dataset, batch_size=32, shuffle=True):
        """Create the DataLoader for validation."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.config, "num_workers", 0),
            pin_memory=True,
        )

    # === Model and Optimizer Construction ===
    def _build_model(self, dataset):
        """Instantiate the Transformer model.
        This is the main model of the Transformer architecture.
        It is a encoder-decoder model.
        The encoder is a stack of Transformer encoder layers.
        The decoder is a stack of Transformer decoder layers.
        The encoder and decoder are connected by an attention mechanism.
        """
        return Transformer(
            input_vocab_size=len(dataset.vocab_en),#input
            output_vocab_size=len(dataset.vocab_zh),#target
            d_model=self.d_model,#lenght of the input vector
            n_heads=self.n_heads,#number of heads in the multi-head attention
            n_layers=self.n_layers,#number of transformer blocks in the encoder and decoder
            d_ff=self.d_ff,#lenght of the feedforward network
            dropout=self.dropout,#dropout rate
            max_seq_len=self.max_length#maximum sequence length
        ).to(self.device)

    def _build_optimizer(self):
        """here we use Adam optimizer as this is a good optimizer for Transformer"""

        return optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_criterion(self):
        """Create the loss function. only compared the output of the model and the target when the target is not the pad id"""
        return nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def _build_scheduler(self):
        """
        Create and return a learning rate scheduler for the optimizer.

        Here we use StepLR, which reduces the learning rate by a factor of 'gamma'
        every 'step_size' epochs. This helps the model converge by lowering the learning
        rate as training progresses.

        Returns:
            torch.optim.lr_scheduler.StepLR: The learning rate scheduler.
        """
        # StepLR is a common scheduler for Transformers. It helps prevent the optimizer
        # from overshooting minima by reducing the learning rate at set intervals.
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
        return scheduler

    # === Training Loop ===

    def _train(self):
        """Main training loop."""
        self.optimizer.zero_grad()
        self.model.train()

        for epoch in trange(self.num_epochs, desc="Epochs"):
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_dataloader, start=1):
                source, target = batch
                source = source.to(self.device)
                target = target.to(self.device)

                # Teacher Forcing: During training, we want the model to learn to predict the next word.
                # To do this, we split the target sequence (target) into two parts:
                # target_in is the part fed into the model (removing the last word),
                # target_labels is what we want the model to predict (removing the first word).
                # This way, at each step, the model uses the real previous words to predict the next word.
                # For example, if target = [BOS, "I", "love", "you", EOS]
                # then target_in = [BOS, "I", "love", "you"]
                #      target_labels = ["I", "love", "you", EOS]
                target_in = target[:, :-1]      # Input to the model (remove last word)
                target_labels = target[:, 1:]   # Expected output (remove first word)


                logits = self.model(source, target_in)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_labels.reshape(-1),
                )
                # TODO: (Label Smoothing) Replace plain CrossEntropy with label-smoothed CE (e.g., epsilon=0.1)
                #       to improve generalization and calibration.
                
                # Setting gradients to None can save memory and may improve performance.
                # However, if you are using the AdamW optimizer, be careful:
                # AdamW's weight decay will skip parameters whose gradients are None,
                # which could lead to incorrect training behavior.
                # For most cases with Adam, this is safe and can be beneficial.
                # For more details, see: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                self.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               getattr(self.config, "max_grad_norm", 1.0))
                self.optimizer.step()

                self.global_step += 1
                epoch_loss += loss.item()

                if self.global_step % self.log_interval == 0:
                    self._log_training_progress(epoch, self.global_step, loss.item())

            # StepLR typically per-epoch
            if isinstance(self.scheduler, optim.lr_scheduler.StepLR):
                self.scheduler.step()
            # TODO: (Noam Scheduler) Optionally replace StepLR with Noam LR schedule
            #       (warmup steps then inverse sqrt decay) for Transformer training.

            avg_loss = epoch_loss / max(1, len(self.train_dataloader))
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

    # === Checkpointing and Logging ===

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model, optimizer, and scheduler states."""
        os.makedirs(self.best_model_path, exist_ok=True)
        name = f"{self.best_model_name}_epoch{epoch}.pt" if not is_best else f"{self.best_model_name}_best.pt"
        ckpt_path = os.path.join(self.best_model_path, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "global_step": self.global_step,
            "config": vars(self.config),
        }, ckpt_path)

    def _log_training_progress(self, epoch: int, step: int, loss: float):
        """Log the training progress."""
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} step={step} loss={loss:.4f} lr={lr:.2e}")

    def _load_checkpoint(self, path: str):
        """Load model, optimizer, and scheduler states from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)

    # === Validation and Prediction ===
    def _validate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                source, target = batch
                source = source.to(self.device)
                target = target.to(self.device)
                target_in = target[:, :-1]
                target_labels = target[:, 1:]
                logits = self.model(source, target_in)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_labels.reshape(-1))
                epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(self.val_dataloader))
        return avg_loss

    @torch.no_grad()
    def _predict_batch(self, src_batch: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        """Generate predictions for a batch of source sentences."""
        self.model.eval()
        src = src_batch.to(self.device)
        bos = self.train_dataset.vocab_zh["<bos>"]
        eos = self.train_dataset.vocab_zh["<eos>"]
        tgt = torch.full((src.size(0), 1), bos, dtype=torch.long, device=self.device)
        for _ in range(max_len):
            logits = self.model(src, tgt)  # [B, t, V]
            # TODO: (Beam Search) Support beam search with length penalty; fall back to greedy.
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_id], dim=1)
            if (next_id == eos).all():
                break
        return tgt

    # === Public API ===

    def run(self):
        """Entry point to start training."""
        self._train()

    def evaluate(self):
        """Evaluate the model on the validation set."""
        return self._validate()

    def predict(self, src_batch: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        """Generate predictions for a batch of source sentences."""
        return self._predict_batch(src_batch, max_len)

if __name__ == "__main__":
    # Example: run with default config dictionary
    from pathlib import Path

    config_dict = {
        "train_dataset_path": Path("data/train.en-zh.txt"),
        "val_dataset_path": Path("data/val.en-zh.txt"),
        "max_length": 32,
        "max_vocab_size": 10000,
        "batch_size": 32,
        "shuffle": True,
        "gpu_id": 0,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "lr": 0.001,
        "step_size": 10,
        "gamma": 0.1,
        "num_epochs": 10,
        "log_interval": 10,
        "save_interval": 10,
        "best_model_path": Path("best_model"),
        "best_model_name": "best_model",
    }

    config = TrainerConfig(**config_dict)
    config.validate()

    trainer = Trainer(config)
    trainer.run()

    
    print("Training finished.")

    print("Best model saved at:", trainer.best_model_path)
    print("Best model epoch:", trainer.best_epoch)
    print("Best model loss:", trainer.best_loss)
    print("Best model global step:", trainer.global_step)
    print("Best model config:", trainer.config)
    print("Best model optimizer:", trainer.optimizer)
    print("Best model scheduler:", trainer.scheduler)
    print("Best model model:", trainer.model)