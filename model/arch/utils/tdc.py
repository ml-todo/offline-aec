from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TdcConfig:
    sample_rate: int = 16000
    max_delay_ms: float = 200.0
    enabled: bool = True

    @property
    def max_delay_samples(self) -> int:
        return int(round(self.sample_rate * self.max_delay_ms / 1000.0))


def gcc_phat_delay(ref: torch.Tensor, mic: torch.Tensor, max_delay: int) -> torch.Tensor:
    """
    Estimate delay (in samples) between ref and mic with GCC-PHAT.
    Args:
      ref, mic: (B, T) float
    Returns:
      delay: (B,) int64, positive means ref should be delayed (shift right) to align with mic.
    """
    # FFT length: next pow2 for speed + enough for circular corr
    B, T = ref.shape
    n = 1
    while n < 2 * T:
        n *= 2

    R = torch.fft.rfft(ref, n=n)
    M = torch.fft.rfft(mic, n=n)
    cross = M * torch.conj(R)
    cross = cross / (torch.abs(cross).clamp_min(1e-8))
    corr = torch.fft.irfft(cross, n=n)

    # Shift to have negative lags at the end
    corr = torch.cat([corr[:, -max_delay:], corr[:, : max_delay + 1]], dim=1)  # (B, 2*max_delay+1)
    idx = torch.argmax(corr, dim=1)
    delay = idx.to(torch.int64) - max_delay
    return delay


def apply_delay(x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
    """
    Shift x by delay samples (per batch element). Positive delay shifts right (pads left with 0).
    Args:
      x: (B, T)
      delay: (B,) int64
    Returns:
      shifted: (B, T)
    """
    B, T = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        d = int(delay[b].item())
        if d == 0:
            out[b] = x[b]
        elif d > 0:
            out[b, d:] = x[b, : T - d]
        else:
            d = -d
            out[b, : T - d] = x[b, d:]
    return out


def tdc_align(ref: torch.Tensor, mic: torch.Tensor, cfg: TdcConfig) -> torch.Tensor:
    """
    Align ref to mic by estimating delay and shifting ref.
    """
    if not cfg.enabled:
        return ref
    max_d = cfg.max_delay_samples
    delay = gcc_phat_delay(ref, mic, max_delay=max_d)
    return apply_delay(ref, delay)



