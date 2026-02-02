from __future__ import annotations
import torch

def resolve_device(device: str = "auto") -> torch.device:
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in {"cuda", "gpu"}:
        return torch.device("cuda")
    return torch.device("cpu")
