import os
import tempfile
from pathlib import Path

import torch
import pytest

from src.training.trainer import Trainer
from src.training.config import TrainerConfig


def _write_tsv(p: Path, pairs):
    with open(p, "w", encoding="utf-8") as f:
        for en, zh in pairs:
            f.write(f"{en}\t{zh}\n")


def test_trainer_init_and_one_epoch(tmp_path: Path):
    # Create tiny train/val files
    train_file = tmp_path / "train.en-zh.txt"
    val_file = tmp_path / "val.en-zh.txt"
    _write_tsv(train_file, [
        ("Hello world", "你好世界"),
        ("Good morning", "早上好"),
        ("How are you", "你好吗"),
        ("Thank you", "谢谢"),
    ])
    _write_tsv(val_file, [
        ("Hello", "你好"),
        ("World", "世界"),
    ])

    out_dir = tmp_path / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small model/config for speed
    cfg = TrainerConfig(
        train_dataset_path=str(train_file),
        val_dataset_path=str(val_file),
        max_length=16,
        max_vocab_size=500,
        batch_size=2,
        shuffle=True,
        gpu_id=0,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.1,
        lr=3e-4,
        step_size=10,
        gamma=0.5,
        num_epochs=1,
        log_interval=10,
        save_interval=1,
        best_model_path=str(out_dir),
        best_model_name="transformer",
    )
    cfg.validate()

    trainer = Trainer(cfg)

    # Sanity on wiring
    assert trainer.train_dataset is not None
    assert trainer.val_dataset is not None
    assert trainer.train_dataloader is not None
    assert trainer.val_dataloader is not None
    assert isinstance(trainer.pad_id, int)

    # Run a very short training (1 epoch)
    trainer.run()

    # Should have advanced steps
    assert trainer.global_step > 0

    # Best checkpoint should exist
    best_ckpt = Path(trainer.best_model_path) / f"{trainer.best_model_name}_best.pt"
    assert best_ckpt.exists()


@pytest.mark.parametrize("max_len", [8, 12])
def test_trainer_validate_and_predict(tmp_path: Path, max_len: int):
    train_file = tmp_path / "train.en-zh.txt"
    val_file = tmp_path / "val.en-zh.txt"
    _write_tsv(train_file, [("I love you", "我爱你"), ("Good night", "晚安")])
    _write_tsv(val_file, [("Hello", "你好")])

    out_dir = tmp_path / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig(
        train_dataset_path=str(train_file),
        val_dataset_path=str(val_file),
        max_length=max_len,
        max_vocab_size=200,
        batch_size=2,
        shuffle=False,
        gpu_id=0,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.1,
        lr=5e-4,
        step_size=5,
        gamma=0.7,
        num_epochs=1,
        log_interval=5,
        save_interval=1,
        best_model_path=str(out_dir),
        best_model_name="transformer",
    )
    cfg.validate()
    trainer = Trainer(cfg)

    # One quick epoch
    trainer.run()

    # Validate returns a finite float
    val_loss = trainer._validate()
    assert isinstance(val_loss, float)
    assert val_loss == val_loss  # not NaN

    # Predict one batch from val
    src_batch, _ = next(iter(trainer.val_dataloader))
    preds = trainer._predict_batch(src_batch, max_len=max_len)
    assert preds.dim() == 2  # [B, T']
    # BOS at position 0
    bos_id = trainer.val_dataset.vocab_zh["<bos>"]
    assert torch.all(preds[:, 0] == bos_id)

