"""
Model-agnostic enhancement interface.

The evaluation server calls `enhance(lpb, mic, checkpoint_path)` to run AEC.
Implement your model here; the server doesn't need to know the internals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000


def _load_model(ckpt_path: str):
    """Load a TSPNN-compatible checkpoint. Replace this with your own model."""
    import sys
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.arch.models import TSPNNBaseline
    from model.arch.utils import StftConfig, stft, istft, TdcConfig, tdc_align

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = TSPNNBaseline(cfg)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        res = model.load_state_dict(ckpt["model"], strict=False)
        vad_prefixes = ("coarse.vad_", "coarse.vad_head", "coarse.vad_conv")
        if not (
            all(k.startswith(vad_prefixes) for k in res.missing_keys)
            and all(k.startswith(vad_prefixes) for k in res.unexpected_keys)
        ):
            raise
    model.eval()
    return model, cfg, (StftConfig, stft, istft, TdcConfig, tdc_align)


_model_cache: dict[str, tuple] = {}


def enhance(lpb: np.ndarray, mic: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """
    Run AEC: cancel echo from mic using lpb reference.

    Args:
        lpb: far-end reference signal, float32, 16 kHz mono
        mic: microphone signal (echo-contaminated), float32, 16 kHz mono
        checkpoint_path: absolute path to a .pt checkpoint

    Returns:
        enhanced signal, float32, 16 kHz mono, same length as min(lpb, mic)
    """
    if checkpoint_path not in _model_cache:
        if len(_model_cache) >= 4:
            _model_cache.pop(next(iter(_model_cache)))
        _model_cache[checkpoint_path] = _load_model(checkpoint_path)

    model, cfg, (StftConfig, stft, istft, TdcConfig, tdc_align) = _model_cache[checkpoint_path]
    n = min(len(lpb), len(mic))
    lpb, mic = lpb[:n], mic[:n]

    st = cfg.get("stft", {})
    stft_cfg = StftConfig(
        sample_rate=SAMPLE_RATE,
        win_ms=float(st.get("win_ms", 20)),
        hop_ms=float(st.get("hop_ms", 10)),
        n_fft=int(st.get("n_fft", 512)),
        center=bool(st.get("center", False)),
        alpha=float(st.get("alpha", 0.3)),
    )
    tc = cfg.get("tdc", {})
    tdc_cfg = TdcConfig(
        sample_rate=SAMPLE_RATE,
        max_delay_ms=float(tc.get("max_delay_ms", 200.0)),
        enabled=bool(tc.get("enabled", False)),
    )

    in_rms = float(np.sqrt(np.mean(np.square(mic)) + 1e-8))
    norm = max(in_rms, 1e-8)

    with torch.no_grad():
        ref = torch.from_numpy(np.asarray(lpb, np.float32)).unsqueeze(0) / norm
        mic_t = torch.from_numpy(np.asarray(mic, np.float32)).unsqueeze(0) / norm
        ref = tdc_align(ref, mic_t, tdc_cfg)
        _, Ofine, _ = model(stft(ref, stft_cfg), stft(mic_t, stft_cfg))
        try:
            out = istft(Ofine, stft_cfg, length=n).squeeze(0).cpu().numpy().astype(np.float32)
        except RuntimeError:
            import librosa
            Z = Ofine.squeeze(0).cpu().numpy()
            out = librosa.istft(
                Z, hop_length=stft_cfg.hop_length, win_length=stft_cfg.win_length,
                window="hann", center=stft_cfg.center, length=n,
            ).astype(np.float32)

    return out * norm
