from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainOnlyBatchNorm2d(nn.BatchNorm2d):
    """
    Apply BN only during training; identity in eval.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.training:
            return x
        return super().forward(x)


class TrainOnlyBatchNorm1d(nn.BatchNorm1d):
    """
    Apply BN only during training; identity in eval.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.training:
            return x
        return super().forward(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: Tuple[int, int], stride: Tuple[int, int], causal: bool = False):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.causal = causal
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=(0, 0))
        self.bn = TrainOnlyBatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        kf, kt = self.kernel
        pf = kf // 2
        pt = kt - 1 if self.causal else kt // 2
        # Pad format: (pad_time_left, pad_time_right, pad_freq_top, pad_freq_bottom)
        return F.pad(x, (pt, 0 if self.causal else pt, pf, pf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        return self.act(self.bn(self.conv(x)))


class TrConv2dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: Tuple[int, int], stride: Tuple[int, int], causal: bool = False):
        super().__init__()
        self.causal = causal
        self.kernel = kernel
        self.stride = stride
        if causal:
            self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            self.conv = Conv2dBlock(in_ch, out_ch, kernel, stride=(1, 1), causal=True)
        else:
            self.deconv = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=kernel, stride=stride, padding=(kernel[0] // 2, kernel[1] // 2), output_padding=(stride[0] - 1, stride[1] - 1)
            )
            self.bn = TrainOnlyBatchNorm2d(out_ch)
            self.act = nn.PReLU(out_ch)

    def _expected_out_size(self, in_hw: Tuple[int, int]) -> Tuple[int, int]:
        """
        Match ConvTranspose2d output size:
          out = in*stride - 2*pad + kernel - 1 (output_padding = stride-1)
        """
        in_h, in_w = in_hw
        kf, kt = self.kernel
        sf, st = self.stride
        pf = kf // 2
        pt = kt // 2
        out_h = in_h * sf - 2 * pf + kf - 1
        out_w = in_w * st - 2 * pt + kt - 1
        return out_h, out_w

    def _match_size(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        th, tw = target_hw
        h, w = x.shape[-2], x.shape[-1]
        if h > th:
            x = x[..., :th, :]
        if w > tw:
            x = x[..., :tw]
        if h < th or w < tw:
            pad_h = max(0, th - x.shape[-2])
            pad_w = max(0, tw - x.shape[-1])
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            target_hw = self._expected_out_size((x.shape[-2], x.shape[-1]))
            x = self.upsample(x)
            y = self.conv(x)
            return self._match_size(y, target_hw)
        return self.act(self.bn(self.deconv(x)))


class GatedTrConv2d(nn.Module):
    """
    Simplified gated transpose conv:
      gate = sigmoid(Wg(enc_skip))
      out  = TrConv(x) * gate
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, kernel: Tuple[int, int], stride: Tuple[int, int], causal: bool = False):
        super().__init__()
        self.tr = TrConv2dBlock(in_ch, out_ch, kernel, stride, causal=causal)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        y = self.tr(x)
        g = self.gate(skip)
        # handle shape mismatch by center crop (due to padding/stride choices)
        if y.shape[-2:] != g.shape[-2:]:
            h = min(y.shape[-2], g.shape[-2])
            w = min(y.shape[-1], g.shape[-1])
            y = y[..., :h, :w]
            g = g[..., :h, :w]
        return y * g


class FTGRULayer(nn.Module):
    """
    Frequency-time GRU:
      - BiGRU over frequency axis (per time frame)
      - UniGRU over time axis

    Input:  (B, C, F, T)
    Output: (B, C_out, F, T)
    """

    def __init__(self, channels: int, f_hidden: int, t_hidden: int):
        super().__init__()
        self.channels = channels
        self.f_gru = nn.GRU(input_size=channels, hidden_size=f_hidden, batch_first=True, bidirectional=True)
        self.t_gru = nn.GRU(input_size=2 * f_hidden, hidden_size=t_hidden, batch_first=True, bidirectional=False)
        self.proj = nn.Linear(t_hidden, channels)
        self.bn = TrainOnlyBatchNorm2d(channels)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Freq, Time = x.shape
        # F-GRU: for each time, run along frequency with feature dim = C
        xf = x.permute(0, 3, 2, 1).contiguous()  # (B, T, F, C)
        xf = xf.view(B * Time, Freq, C)          # (B*T, F, C)
        yf, _ = self.f_gru(xf)                   # (B*T, F, 2*f_hidden)

        # T-GRU: for each frequency bin, run along time
        yt = yf.view(B, Time, Freq, -1).permute(0, 2, 1, 3).contiguous()  # (B, F, T, 2*f_hidden)
        yt = yt.view(B * Freq, Time, -1)                                  # (B*F, T, 2*f_hidden)
        yt, _ = self.t_gru(yt)                                             # (B*F, T, t_hidden)
        yt = self.proj(yt)                                                 # (B*F, T, C)
        yt = yt.view(B, Freq, Time, C).permute(0, 3, 1, 2).contiguous()    # (B, C, F, T)
        return self.act(self.bn(yt))



