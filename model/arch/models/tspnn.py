from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.arch.models.blocks import Conv2dBlock, FTGRULayer, GatedTrConv2d, TrainOnlyBatchNorm1d, TrainOnlyBatchNorm2d


@dataclass(frozen=True)
class DeepFilterConfig:
    nf: int = 3
    nt: int = 3
    nl: int = 1


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # both complex
    return a * b


def _resize_2d_like(x: torch.Tensor, ref_hw: tuple[int, int]) -> torch.Tensor:
    """
    Resize NCHW tensor to match (H, W) = (F, T) using interpolation.
    """
    if x.shape[-2:] == ref_hw:
        return x
    return F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)


def make_complex_mask(mask_2ch: torch.Tensor) -> torch.Tensor:
    """
    Convert a 2-channel real mask into complex.
    Args:
      mask_2ch: (B, 2, F, T) float
    Returns:
      mask: (B, F, T) complex
    """
    mr = mask_2ch[:, 0]
    mi = mask_2ch[:, 1]
    return torch.complex(mr, mi)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, channels: List[int], kernels: List[Tuple[int, int]], strides: List[Tuple[int, int]], causal: bool = False):
        super().__init__()
        assert len(channels) == len(kernels) == len(strides)
        blocks = []
        ch_in = in_ch
        for ch_out, k, s in zip(channels, kernels, strides):
            blocks.append(Conv2dBlock(ch_in, ch_out, tuple(k), tuple(s), causal=causal))
            ch_in = ch_out
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        skips = []
        for b in self.blocks:
            x = b(x)
            skips.append(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, channels: List[int], kernels: List[Tuple[int, int]], strides: List[Tuple[int, int]], causal: bool = False):
        super().__init__()
        # channels are encoder channels; decoder traverses reversed
        rev_ch = list(channels)[::-1]
        rev_k = list(kernels)[::-1]
        rev_s = list(strides)[::-1]
        blocks = []
        for i in range(len(rev_ch) - 1):
            in_ch = rev_ch[i]
            out_ch = rev_ch[i + 1]
            skip_ch = rev_ch[i]  # skip at same resolution
            blocks.append(GatedTrConv2d(in_ch, out_ch, skip_ch=skip_ch, kernel=tuple(rev_k[i]), stride=tuple(rev_s[i]), causal=causal))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # skips are encoder outputs; use reversed alignment
        rev_skips = list(skips)[::-1]
        for i, b in enumerate(self.blocks):
            skip = rev_skips[i]
            x = b(x, skip)
        return x


class CoarseStage(nn.Module):
    def __init__(self, in_ch: int, enc_channels, enc_kernels, enc_strides, fgru_hidden: int, tgru_hidden: int, causal: bool = True):
        super().__init__()
        self.encoder = Encoder(in_ch, enc_channels, enc_kernels, enc_strides, causal=causal)
        bottleneck_ch = enc_channels[-1]
        self.ftgru1 = FTGRULayer(bottleneck_ch, f_hidden=fgru_hidden, t_hidden=tgru_hidden)
        self.ftgru2 = FTGRULayer(bottleneck_ch, f_hidden=fgru_hidden, t_hidden=tgru_hidden // 2 if tgru_hidden > 1 else tgru_hidden)
        self.decoder = Decoder(enc_channels, enc_kernels, enc_strides, causal=causal)
        # predict complex mask (real+imag), sigmoid per paper
        self.mask_head = nn.Sequential(
            nn.Conv2d(enc_channels[0], 2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        # VAD path per paper: Conv2D -> F-GRU -> Conv1D -> Conv1D
        self.vad_conv2d = nn.Conv2d(bottleneck_ch, 16, kernel_size=1, stride=1, padding=0)
        self.vad_bn2d = TrainOnlyBatchNorm2d(16)
        self.vad_act2d = nn.PReLU(16)
        self.vad_fgru = nn.GRU(input_size=16, hidden_size=8, batch_first=True, bidirectional=True)
        self.vad_conv1d_1 = nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0)
        self.vad_bn1d = TrainOnlyBatchNorm1d(16)
        self.vad_act1d = nn.PReLU(16)
        self.vad_conv1d_2 = nn.Conv1d(16, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, feats_2ch: torch.Tensor):
        """
        Args:
          feats_2ch: (B, 2, F, T) float
        Returns:
          mask: (B, 2, F, T) float
          vad_logits: (B, 2, T) float
        """
        x = feats_2ch
        x, skips = self.encoder(x)
        x = self.ftgru1(x)
        x = self.ftgru2(x)

        # VAD head from bottleneck features (B, C, F', T) -> (B, 2, T)
        v = self.vad_act2d(self.vad_bn2d(self.vad_conv2d(x)))  # (B, 16, F', T)
        B, C, Freq, Time = v.shape
        vf = v.permute(0, 3, 2, 1).contiguous().view(B * Time, Freq, C)  # (B*T, F, C)
        _, h_n = self.vad_fgru(vf)  # h_n: (2, B*T, 8)
        v = h_n.permute(1, 0, 2).contiguous().view(B, Time, -1).permute(0, 2, 1)  # (B, 16, T)
        v = self.vad_act1d(self.vad_bn1d(self.vad_conv1d_1(v)))
        vad_logits = self.vad_conv1d_2(v)  # (B, 2, T)

        x = self.decoder(x, skips)
        m = self.mask_head(x)
        # Ensure mask matches input TF resolution (decoder stride/padding can differ across impls)
        m = _resize_2d_like(m, feats_2ch.shape[-2:])
        return m, vad_logits


class FineStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        enc_channels,
        enc_kernels,
        enc_strides,
        fgru_hidden: int,
        tgru_hidden: int,
        df_cfg: DeepFilterConfig,
        causal: bool = True,
    ):
        super().__init__()
        self.df_cfg = df_cfg
        self.encoder = Encoder(in_ch, enc_channels, enc_kernels, enc_strides, causal=causal)
        bottleneck_ch = enc_channels[-1]
        self.ftgru1 = FTGRULayer(bottleneck_ch, f_hidden=fgru_hidden, t_hidden=tgru_hidden)
        self.ftgru2 = FTGRULayer(bottleneck_ch, f_hidden=fgru_hidden, t_hidden=max(1, tgru_hidden // 2))
        self.decoder = Decoder(enc_channels, enc_kernels, enc_strides, causal=causal)
        # Deep-filter mask head: output complex weights per neighbor (real+imag per tap)
        taps = (2 * df_cfg.nf + 1) * (df_cfg.nt + df_cfg.nl + 1)
        self.mask_head = nn.Conv2d(enc_channels[0], 2 * taps, kernel_size=1, stride=1, padding=0)

    def forward(self, feats_3ch: torch.Tensor) -> torch.Tensor:
        """
        Args:
          feats_3ch: (B, 3, F, T) float
        Returns:
          df_mask: (B, 2*taps, F, T) float (real/imag per tap)
        """
        x = feats_3ch
        x, skips = self.encoder(x)
        x = self.ftgru1(x)
        x = self.ftgru2(x)
        x = self.decoder(x, skips)
        m = self.mask_head(x)
        # Keep DF mask resolution aligned to input TF resolution
        m = _resize_2d_like(m, feats_3ch.shape[-2:])
        return m


def deep_filter_apply(coarse: torch.Tensor, df_mask: torch.Tensor, df_cfg: DeepFilterConfig) -> torch.Tensor:
    """
    Apply deep filter mask to coarse STFT.
    Args:
      coarse: (B, F, T) complex
      df_mask: (B, 2*taps, F, T) float
    Returns:
      fine: (B, F, T) complex
    """
    B, Freq, Time = coarse.shape
    nf, nt, nl = df_cfg.nf, df_cfg.nt, df_cfg.nl
    taps = (2 * nf + 1) * (nt + nl + 1)
    assert df_mask.shape[1] == 2 * taps

    # Build padded real/imag separately (since torch pad doesn't support complex well for all modes)
    cr = coarse.real
    ci = coarse.imag
    crp = F.pad(cr, (nt, nl, nf, nf))  # pad time then freq
    cip = F.pad(ci, (nt, nl, nf, nf))

    # Collect neighborhood patches
    # For each (f, t), gather (2*nf+1) * (nt+nl+1) points from padded tensors.
    patches_r = []
    patches_i = []
    for df in range(-nf, nf + 1):
        for dt in range(-nt, nl + 1):
            fr0 = nf + df
            tr0 = nt + dt
            patches_r.append(crp[:, fr0 : fr0 + Freq, tr0 : tr0 + Time])
            patches_i.append(cip[:, fr0 : fr0 + Freq, tr0 : tr0 + Time])
    # (taps, B, F, T) -> (B, taps, F, T)
    pr = torch.stack(patches_r, dim=0).permute(1, 0, 2, 3)
    pi = torch.stack(patches_i, dim=0).permute(1, 0, 2, 3)

    # mask: (B, 2*taps, F, T) -> (B, taps, F, T) complex
    mr = df_mask[:, 0::2]
    mi = df_mask[:, 1::2]

    # complex dot product: sum_k (p_k * m_k)
    out_r = torch.sum(pr * mr - pi * mi, dim=1)
    out_i = torch.sum(pr * mi + pi * mr, dim=1)
    return torch.complex(out_r, out_i)


class TSPNNBaseline(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        stft_cfg = cfg["stft"]
        self.alpha = float(stft_cfg.get("alpha", 0.3))

        c = cfg["model"]["coarse"]
        f = cfg["model"]["fine"]
        df = f["deep_filter"]
        self.df_cfg = DeepFilterConfig(nf=int(df["nf"]), nt=int(df["nt"]), nl=int(df["nl"]))

        self.coarse = CoarseStage(
            in_ch=2,
            enc_channels=c["encoder_channels"],
            enc_kernels=c["encoder_kernels"],
            enc_strides=c["encoder_strides"],
            fgru_hidden=int(c["fgru_hidden"]),
            tgru_hidden=int(c["tgru_hidden"]),
            causal=True,
        )
        self.fine = FineStage(
            in_ch=3,
            enc_channels=f["encoder_channels"],
            enc_kernels=f["encoder_kernels"],
            enc_strides=f["encoder_strides"],
            fgru_hidden=int(f["fgru_hidden"]),
            tgru_hidden=int(f["tgru_hidden"]),
            df_cfg=self.df_cfg,
            causal=True,
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
          X: (B, F, T) complex  - reference STFT (aligned)
          Y: (B, F, T) complex  - mic STFT
        Returns:
          Ocoarse, Ofine: complex STFTs
        """
        feats_c = torch.stack([torch.abs(X).clamp_min(1e-8) ** self.alpha, torch.abs(Y).clamp_min(1e-8) ** self.alpha], dim=1)
        m2, vad_logits = self.coarse(feats_c)
        Mc = make_complex_mask(m2)
        Ocoarse = complex_mul(Y, Mc)

        feats_f = torch.stack(
            [
                torch.abs(X).clamp_min(1e-8) ** self.alpha,
                torch.abs(Y).clamp_min(1e-8) ** self.alpha,
                torch.abs(Ocoarse).clamp_min(1e-8) ** self.alpha,
            ],
            dim=1,
        )
        df_mask = self.fine(feats_f)
        # Ensure df_mask matches Ocoarse TF resolution (safety)
        if df_mask.shape[-2:] != Ocoarse.shape[-2:]:
            df_mask = _resize_2d_like(df_mask, Ocoarse.shape[-2:])
        Ofine = deep_filter_apply(Ocoarse, df_mask, self.df_cfg)
        return Ocoarse, Ofine, vad_logits


