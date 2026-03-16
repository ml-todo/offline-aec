from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int = 16000
    win_ms: float = 20.0
    hop_ms: float = 10.0
    n_fft: int = 512
    center: bool = False
    alpha: float = 0.3

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))


def hann_window(win_length: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(win_length, periodic=True, device=device)


def stft(x: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """
    Args:
      x: (B, T) float32
    Returns:
      X: (B, F, frames) complex64
    """
    window = hann_window(cfg.win_length, x.device)
    X = torch.stft(
        x,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        return_complex=True,
    )
    return X


def istft(X: torch.Tensor, cfg: StftConfig, length: int) -> torch.Tensor:
    window = hann_window(cfg.win_length, X.device)
    x = torch.istft(
        X,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=cfg.center,
        length=length,
    )
    return x


def compress_mag(X: torch.Tensor, alpha: float) -> torch.Tensor:
    # |X|^alpha
    mag = torch.abs(X).clamp_min(1e-8)
    return mag ** alpha


def stack_features(ref: torch.Tensor, mic: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Args:
      ref, mic: (B, F, T) complex
    Returns:
      feats: (B, 2, F, T) float
    """
    xr = compress_mag(ref, alpha)
    ym = compress_mag(mic, alpha)
    return torch.stack([xr, ym], dim=1)


def stack_features_fine(ref: torch.Tensor, mic: torch.Tensor, coarse: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Args:
      ref, mic, coarse: (B, F, T) complex
    Returns:
      feats: (B, 3, F, T) float
    """
    xr = compress_mag(ref, alpha)
    ym = compress_mag(mic, alpha)
    oc = compress_mag(coarse, alpha)
    return torch.stack([xr, ym, oc], dim=1)


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor along dim to make length divisible by multiple.
    Returns padded tensor and amount padded at end.
    """
    length = x.size(dim)
    rem = length % multiple
    if rem == 0:
        return x, 0
    pad_amt = multiple - rem
    pad = [0, 0] * (x.dim())
    # torch.nn.functional.pad expects last dims in reverse order; we only pad along dim=-1 usage here.
    if dim != -1:
        raise ValueError("pad_to_multiple currently supports dim=-1 only")
    y = F.pad(x, (0, pad_amt))
    return y, pad_amt


def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))



