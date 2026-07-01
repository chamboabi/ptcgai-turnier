"""Build a MyModel from config, loading the latest checkpoint if one exists."""

from __future__ import annotations

from pathlib import Path

import torch

import config as cfg
from model import MyModel

DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "out" / "model_best.pth"


def build_model(checkpoint: Path | str | None = DEFAULT_CHECKPOINT) -> MyModel:
    m = MyModel(
        cfg.model["d_model"],
        cfg.model["num_heads"],
        cfg.model["d_feedforward"],
        cfg.model["num_layers_encoder"],
        cfg.model["num_layers_decoder"],
    )
    ckpt = Path(checkpoint) if checkpoint is not None else None
    if ckpt is not None and ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded checkpoint {ckpt}")
    else:
        print("No checkpoint — using randomly initialised model (play is weak).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(device).eval()
