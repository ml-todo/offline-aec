#!/usr/bin/env python3
"""
AEC Evaluation Server: TSPNN (offline), DTLN-aec, WebRTC AEC3, and our model (online).

Serves the comparison UI and APIs for enhancement and AECMOS scoring.
"""

from __future__ import annotations

import argparse
import glob
import http.server
import io
import json
import math
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.parse
import wave
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

PORT = 8888
SAMPLE_RATE = 16000

# AECMOS mel spectrogram (must match TSPNN eval)
AECMOS_MEL_N_FFT = 513
AECMOS_MEL_HOP = 256
AECMOS_MEL_N_MELS = 160
AECMOS_MAX_SCORE_SECONDS = 20
AECMOS_PAD_FRAMES = 20  # frames to pad for ONNX input

# AECMOS ONNX model (optional): scoring returns 503 until this file exists.
AECMOS_MODEL_NAME = "Run_1657188842_Stage_0.onnx"
AECMOS_HINT = (
    "Download Run_1657188842_Stage_0.onnx from Microsoft AEC-Challenge (AECMOS) "
    "and place it in external/TSPNN/eval/"
)

ENHANCE_HINTS = {
    "dtln": "pip install tensorflow",
    "webrtc_aec3": "cd evaluation/webrtc-aec3-node && npm install; ensure Node.js is installed",
    "ckpt": "pip install -r evaluation/requirements.txt",
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def _path_inside(base: Path, resolved: Path) -> bool:
    """True if resolved is under base (no traversal outside base)."""
    base_str = str(base.resolve()) + os.sep
    return str(resolved).startswith(base_str)


def _tspnn_root() -> Path:
    return _project_root() / "external" / "TSPNN"


def _results_root() -> Path:
    return _tspnn_root() / "results"


def _aecmos_model_path() -> Path:
    return _tspnn_root() / "eval" / AECMOS_MODEL_NAME


DTLN_DEFAULT_ROOT = (_project_root() / "external" / "DTLN-aec").resolve()
DTLN_DEFAULT_MODEL = (
    DTLN_DEFAULT_ROOT / "pretrained_models" / "dtln_aec_512"
).resolve()


# ---------------------------------------------------------------------------
# Module-level caches (handler is instantiated per-request, so lru_cache on
# instance methods would never hit; keep caches here instead)
# ---------------------------------------------------------------------------

_ort_session_cache = None

# Serialize heavy operations (AECMOS, DTLN, ckpt) to avoid OOM when multiple
# requests run in parallel (ThreadingTCPServer).
_heavy_lock = threading.Lock()


def _get_ort_session():
    global _ort_session_cache
    if _ort_session_cache is not None:
        return _ort_session_cache
    import onnxruntime as ort
    model_path = _aecmos_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"AECMOS model not found: {model_path}. {AECMOS_HINT}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # ERROR only; suppresses "Unknown CPU vendor" etc.
    _ort_session_cache = ort.InferenceSession(str(model_path), opts)
    return _ort_session_cache


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _read_wav_16k(wav_path: Path) -> np.ndarray:
    """Load mono WAV at 16 kHz as float32 in [-1, 1]."""
    import librosa
    sig, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    return sig


def _wav_bytes_pcm16(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 [-1,1] audio to 16-bit PCM WAV bytes (mono)."""
    x = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson correlation between two equal-length arrays; None if empty or length mismatch."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if len(a) != len(b) or len(a) == 0:
        return None
    a, b = a - np.mean(a), b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def _rms(x: np.ndarray) -> float:
    """RMS level of array."""
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _match_level(enh: np.ndarray, ref: np.ndarray, max_gain: float = 10.0) -> np.ndarray:
    """Scale enh so its RMS matches ref (for comparable waveform/playback level)."""
    rms_ref = _rms(ref)
    rms_enh = _rms(enh)
    if rms_enh <= 0:
        return enh
    gain = rms_ref / rms_enh
    gain = min(gain, max_gain)
    return (np.asarray(enh, dtype=np.float32) * gain).astype(np.float32)


def _parse_qs(path: str) -> dict[str, list[str]]:
    parsed = urllib.parse.urlparse(path)
    return urllib.parse.parse_qs(parsed.query)


def _qval(q: dict[str, list[str]], key: str, default: Any = None) -> Any:
    return (q.get(key) or [default])[0]


def _safe_segment(name: str, allowed: str = r"[\w\-.]") -> bool:
    """Return True if segment is safe for path components (no path traversal)."""
    return bool(name) and re.match(f"^{allowed}+$", name) and ".." not in name


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path: str) -> str:
        path = urllib.parse.unquote(path).split("?", 1)[0].split("#", 1)[0]
        path = path.lstrip("/")
        root = _project_root()
        if path.startswith("results/"):
            sub = path[len("results/"):]
            if ".." in sub:
                return str(root)
            res = (_results_root() / sub).resolve()
            return str(res) if _path_inside(_results_root(), res) else str(root)
        if path.startswith("evaluation/"):
            sub = path[len("evaluation/"):]
            if ".." in sub:
                return str(root)
            ev_dir = Path(__file__).parent.resolve()
            res = (ev_dir / sub).resolve()
            return str(res) if _path_inside(ev_dir, res) else str(root)
        if not path or ".." in path:
            return str(root)
        res = (root / path).resolve()
        return str(res) if _path_inside(root, res) else str(root)

    # ---- JSON response helper ----

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, audio_f32: np.ndarray):
        wav_bytes = _wav_bytes_pcm16(audio_f32)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)

    # ---- Audio loading helpers ----

    def _load_sample_pair(
        self, scenario: str, sample: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
        if not _safe_segment(scenario) or not _safe_segment(sample):
            return None, None, ["Invalid scenario or sample (path safety)"]
        blind = _results_root() / "blind_test_set_interspeech2021"
        try:
            lpb_path = (blind / scenario / f"{sample}_lpb.wav").resolve()
            mic_path = (blind / scenario / f"{sample}_mic.wav").resolve()
            if not _path_inside(blind, lpb_path) or not _path_inside(blind, mic_path):
                return None, None, ["Path safety check failed"]
        except (ValueError, OSError):
            return None, None, ["Invalid path"]
        missing = [str(p) for p in [lpb_path, mic_path] if not p.exists()]
        if missing:
            return None, None, missing
        lpb = _read_wav_16k(lpb_path)
        mic = _read_wav_16k(mic_path)
        n = min(len(lpb), len(mic))
        return lpb[:n], mic[:n], None

    # ---- Enhancement backends ----

    @staticmethod
    def _dtln_root() -> Path:
        env = os.environ.get("DTLN_ROOT")
        return Path(env).expanduser().resolve() if env else DTLN_DEFAULT_ROOT

    @staticmethod
    def _dtln_model_base(dtln_root: Path) -> Path:
        env = os.environ.get("DTLN_MODEL")
        if env:
            return Path(env).expanduser().resolve()
        if DTLN_DEFAULT_MODEL.exists():
            return DTLN_DEFAULT_MODEL
        candidates = list(dtln_root.glob("pretrained_models/**/dtln_aec_*_1.tflite"))
        if not candidates:
            raise FileNotFoundError("DTLN model not found. Set DTLN_MODEL env var.")
        return Path(str(candidates[0]).replace("_1.tflite", ""))

    def _run_dtln(self, lpb_sig, mic_sig, sample_name: str):
        import soundfile as sf

        dtln_root = self._dtln_root()
        run_py = dtln_root / "run_aec.py"
        if not run_py.exists():
            raise FileNotFoundError(
                f"DTLN-aec not found at {run_py}. Run: git submodule update --init external/DTLN-aec"
            )
        model_base = self._dtln_model_base(dtln_root)

        tmp_root = _project_root() / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dtln_", dir=str(tmp_root)) as work:
            in_dir = Path(work) / "in"
            out_dir = Path(work) / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            sf.write(str(in_dir / f"{sample_name}_mic.wav"), np.asarray(mic_sig, dtype=np.float32), SAMPLE_RATE)
            sf.write(str(in_dir / f"{sample_name}_lpb.wav"), np.asarray(lpb_sig, dtype=np.float32), SAMPLE_RATE)

            python = os.environ.get("DTLN_PYTHON", sys.executable)
            timeout_s = int(os.environ.get("DTLN_TIMEOUT", "120"))
            proc = subprocess.run(
                [python, str(run_py), "-i", str(in_dir), "-o", str(out_dir), "-m", str(model_base)],
                capture_output=True, text=True, timeout=timeout_s,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"DTLN-aec failed: {proc.stderr}")

            out_path = out_dir / f"{sample_name}_mic.wav"
            if not out_path.exists():
                raise FileNotFoundError(f"DTLN output missing: {out_path}")
            import librosa
            enh, _ = librosa.load(str(out_path), sr=SAMPLE_RATE)
            n = min(len(mic_sig), len(enh))
            return _match_level(enh[:n], mic_sig[:n])

    def _run_webrtc_aec3(self, lpb_sig, mic_sig, sample_name: str):
        """Run WebRTC AEC3 via Node (ennuicastr/webrtcaec3.js). Requires npm install in evaluation/webrtc-aec3-node."""
        import soundfile as sf
        run_js = _project_root() / "evaluation" / "webrtc-aec3-node" / "run_aec3.js"
        node_dir = _project_root() / "evaluation" / "webrtc-aec3-node"
        if not run_js.exists():
            raise FileNotFoundError(
                f"AEC3 runner not found: {run_js}. Add evaluation/webrtc-aec3-node/run_aec3.js and run npm install there."
            )
        tmp_root = _project_root() / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        timeout_s = int(os.environ.get("WEBRTC_AEC3_TIMEOUT", "60"))
        node = os.environ.get("WEBRTC_AEC3_NODE", "node")
        with tempfile.TemporaryDirectory(prefix="webrtc_aec3_", dir=str(tmp_root)) as work:
            work_p = Path(work)
            lpb_path = work_p / "lpb.wav"
            mic_path = work_p / "mic.wav"
            out_path = work_p / "out.wav"
            sf.write(str(lpb_path), np.asarray(lpb_sig, dtype=np.float32), SAMPLE_RATE)
            sf.write(str(mic_path), np.asarray(mic_sig, dtype=np.float32), SAMPLE_RATE)
            proc = subprocess.run(
                [node, str(run_js), "--lpb", str(lpb_path), "--mic", str(mic_path), "--out", str(out_path)],
                capture_output=True, text=True, timeout=timeout_s, cwd=str(node_dir),
            )
            if proc.returncode != 0:
                err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
                raise RuntimeError(f"WebRTC AEC3 failed (exit {proc.returncode}): {err or '(no output)'}")
            if not out_path.exists():
                raise FileNotFoundError(f"AEC3 output missing: {out_path}")
            enh = _read_wav_16k(out_path)
            n = min(len(mic_sig), len(enh))
            return _match_level(enh[:n], mic_sig[:n])

    def _enhance_with_ckpt(self, ckpt_path: Path, lpb_sig, mic_sig):
        """Run checkpoint enhancement in a subprocess so OOM kills only the child; server stays up."""
        import soundfile as sf
        run_py = _project_root() / "evaluation" / "run_ckpt_enhance.py"
        if not run_py.exists():
            raise FileNotFoundError(f"Runner not found: {run_py}")
        tmp_root = _project_root() / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        timeout_s = int(os.environ.get("CKPT_ENHANCE_TIMEOUT", "120"))
        with tempfile.TemporaryDirectory(prefix="ckpt_enhance_", dir=str(tmp_root)) as work:
            work_p = Path(work)
            lpb_path = work_p / "lpb.wav"
            mic_path = work_p / "mic.wav"
            out_path = work_p / "out.wav"
            sf.write(str(lpb_path), np.asarray(lpb_sig, dtype=np.float32), SAMPLE_RATE)
            sf.write(str(mic_path), np.asarray(mic_sig, dtype=np.float32), SAMPLE_RATE)
            python = os.environ.get("CKPT_PYTHON", sys.executable)
            proc = subprocess.run(
                [python, str(run_py), "--lpb", str(lpb_path), "--mic", str(mic_path), "--ckpt", str(ckpt_path), "--out", str(out_path)],
                capture_output=True, text=True, timeout=timeout_s, cwd=str(_project_root()),
            )
            if proc.returncode != 0:
                err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
                if proc.returncode == -9:
                    err = err or "Subprocess was killed (exit -9 = SIGKILL). Often OOM: set CKPT_MAX_LEN_S=30 (or add ?max_len_s=30) for short clips, or use a machine with more RAM for full length."
                raise RuntimeError(
                    f"Checkpoint enhancement failed (exit {proc.returncode}): {err or '(no output)'}"
                )
            if not out_path.exists():
                raise FileNotFoundError(f"Enhancement output missing: {out_path}")
            enh = _read_wav_16k(out_path)
            n = min(len(mic_sig), len(enh))
            return _match_level(enh[:n], mic_sig[:n])

    # ---- AECMOS scoring ----

    @staticmethod
    def _mel_transform(sample: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=sample, sr=sr,
            n_fft=AECMOS_MEL_N_FFT, hop_length=AECMOS_MEL_HOP, n_mels=AECMOS_MEL_N_MELS,
        )
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T

    @staticmethod
    def _compute_erle(echo_data: np.ndarray, error_data: np.ndarray) -> float:
        if len(echo_data) == 0 or len(error_data) == 0:
            return 0.0
        echo_ms = float(np.mean(np.square(echo_data)))
        error_ms = float(np.mean(np.square(error_data)))
        if error_ms == 0:
            return 50.0
        return float(10 * math.log10(echo_ms / error_ms))

    def _score_aecmos(self, talk_type: str, lpb_sig, mic_sig, enh_sig):
        if talk_type not in {"nst", "st", "dt"}:
            raise ValueError("talk_type must be one of: nst, st, dt")

        sr = SAMPLE_RATE
        max_samples = AECMOS_MAX_SCORE_SECONDS * sr
        if len(lpb_sig) >= max_samples:
            lpb_sig, mic_sig, enh_sig = lpb_sig[:max_samples], mic_sig[:max_samples], enh_sig[:max_samples]

        lpb_feat = self._mel_transform(lpb_sig, sr)
        mic_feat = self._mel_transform(mic_sig, sr)
        enh_feat = self._mel_transform(enh_sig, sr)

        ne_st = 1 if talk_type == "nst" else 0
        fe_st = 1 if talk_type == "st" else 0

        pad = AECMOS_PAD_FRAMES

        def pad_feat(feat: np.ndarray, flag_val: float) -> np.ndarray:
            return np.concatenate(
                (feat, np.ones((pad, feat.shape[1])) * flag_val, np.zeros((pad, feat.shape[1]))),
                axis=0,
            )

        mic_feat = pad_feat(mic_feat, 1 - fe_st)
        lpb_feat = pad_feat(lpb_feat, 1 - ne_st)
        enh_feat = pad_feat(enh_feat, 1)

        feats = np.expand_dims(np.stack((lpb_feat, mic_feat, enh_feat)).astype(np.float32), axis=0)
        sess = _get_ort_session()
        h0 = np.zeros((4, 1, 64), dtype=np.float32)
        result = sess.run([], {sess.get_inputs()[0].name: feats, "h0": h0})[0]
        return {"echo_mos": float(result[0]), "deg_mos": float(result[1])}

    @staticmethod
    def _scenario_to_talk_type(scenario: str) -> str:
        if "double" in scenario:
            return "dt"
        if "farend" in scenario:
            return "st"
        if "nearend" in scenario or "earphone" in scenario:
            return "nst"
        raise ValueError(f"Unknown scenario: {scenario}")

    @staticmethod
    def _safe_ckpt_path(ckpt: str) -> Path:
        root = _project_root()
        p = (root / ckpt).resolve() if not os.path.isabs(ckpt) else Path(ckpt).resolve()
        if not _path_inside(root, p):
            raise ValueError("Checkpoint path must be inside the project directory.")
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    # ---- Debug JSON for enhancement endpoints ----

    @staticmethod
    def _debug_stats(scenario, sample, lpb, mic, enh, **extra):
        return {
            "scenario": scenario, "sample": sample,
            "rms_lpb": _rms(lpb), "rms_mic": _rms(mic), "rms_enh": _rms(enh),
            "corr_enh_mic": _corr(enh, mic), "corr_enh_lpb": _corr(enh, lpb),
            **extra,
        }

    # ---- GET routes ----

    def do_GET(self):
        try:
            if self.path in ("/", ""):
                self.send_response(302)
                self.send_header("Location", "/evaluation/index.html")
                self.end_headers()
                return

            if self.path.startswith("/api/checkpoints"):
                return self._api_checkpoints()
            if self.path.startswith("/api/enhance_dtln_sample"):
                return self._api_enhance("dtln")
            if self.path.startswith("/api/enhance_webrtc_aec3_sample"):
                return self._api_enhance("webrtc_aec3")
            if self.path.startswith("/api/enhance_sample"):
                return self._api_enhance("ckpt")
            if self.path.startswith("/api/out_sample"):
                return self._api_out_sample()
            if self.path.startswith("/api/aecmos_sample"):
                return self._api_aecmos_sample()

            return super().do_GET()
        except Exception as e:
            print(f"[do_GET] {self.path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            try:
                self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})
            except Exception:
                pass

    def _api_checkpoints(self):
        try:
            root = _project_root()
            patterns = [
                str(root / "model" / "runs" / "**" / "ckpt_*.pt"),
                str(_tspnn_root() / "runs" / "**" / "ckpt_*.pt"),
            ]
            all_ckpts = set()
            for pat in patterns:
                all_ckpts.update(glob.glob(pat, recursive=True))
            rel = []
            for p in sorted(all_ckpts):
                try:
                    rel.append(str(Path(p).resolve().relative_to(root)))
                except ValueError:
                    pass
            return self._send_json(200, {"checkpoints": rel})
        except Exception as e:
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def _api_enhance(self, method: str):
        try:
            q = _parse_qs(self.path)
            scenario = _qval(q, "scenario")
            sample = _qval(q, "sample")
            debug = _qval(q, "debug", "0") == "1"
            if not scenario or not sample:
                return self._send_json(400, {"error": "scenario and sample required"})

            lpb, mic, missing = self._load_sample_pair(scenario, sample)
            if missing:
                return self._send_json(404, {"error": "Missing wav files", "missing": missing})

            if method == "dtln":
                with _heavy_lock:
                    enh = self._run_dtln(lpb, mic, sample)
            elif method == "webrtc_aec3":
                with _heavy_lock:
                    enh = self._run_webrtc_aec3(lpb, mic, sample)
            else:
                ckpt = _qval(q, "ckpt", "model/runs/full_aec_synth/ckpt_final.pt")
                # Subprocess runs in separate process; default 30s to avoid OOM. Set CKPT_MAX_LEN_S=0 for full length.
                max_len_s = _qval(q, "max_len_s") or os.environ.get("CKPT_MAX_LEN_S", "30")
                try:
                    n = int(float(max_len_s) * SAMPLE_RATE)
                    if n > 0:
                        lpb, mic = lpb[:n], mic[:n]
                except Exception:
                    pass
                ckpt_path = self._safe_ckpt_path(ckpt)
                with _heavy_lock:
                    enh = self._enhance_with_ckpt(ckpt_path, lpb, mic)

            if debug:
                extra = {"ckpt": str(ckpt_path)} if method == "ckpt" else {}
                return self._send_json(200, self._debug_stats(scenario, sample, lpb, mic, enh, **extra))

            return self._send_wav(enh)

        except ModuleNotFoundError as err:
            return self._send_json(
                503,
                {"error": str(err), "hint": ENHANCE_HINTS.get(method, ""), "unavailable": True},
            )
        except Exception as e:
            print(f"[enhance {method}] 500: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def _api_out_sample(self):
        """Serve TSPNN (ours) output WAV level-matched to mic so waveform/playback is comparable to input."""
        try:
            q = _parse_qs(self.path)
            scenario = _qval(q, "scenario")
            sample = _qval(q, "sample")
            if not scenario or not sample:
                return self._send_json(400, {"error": "scenario and sample required"})
            lpb, mic, missing = self._load_sample_pair(scenario, sample)
            if missing:
                return self._send_json(404, {"error": "Missing wav files", "missing": missing})
            out_path = _results_root() / "output" / "ours" / scenario / f"{sample}_mic.wav"
            if not out_path.exists():
                return self._send_json(404, {"error": "Missing TSPNN output", "missing": [str(out_path)]})
            enh = _read_wav_16k(out_path)
            n = min(len(mic), len(enh))
            out_matched = _match_level(enh[:n], mic[:n])
            return self._send_wav(out_matched)
        except Exception as e:
            print(f"[out_sample] 500: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def _api_aecmos_sample(self):
        try:
            q = _parse_qs(self.path)
            scenario = _qval(q, "scenario")
            sample = _qval(q, "sample")
            enh_src = _qval(q, "enh", "ours")
            if enh_src not in {"ours", "coarse_stage"}:
                return self._send_json(400, {"error": "enh must be ours or coarse_stage"})

            lpb, mic, missing = self._load_sample_pair(scenario, sample)
            if missing:
                return self._send_json(404, {"error": "Missing wav files", "missing": missing})

            enh_path = _results_root() / "output" / enh_src / scenario / f"{sample}_mic.wav"
            if not enh_path.exists():
                return self._send_json(404, {"error": "Missing enhanced wav", "missing": [str(enh_path)]})

            with _heavy_lock:
                enh_sig = _read_wav_16k(enh_path)
                n = min(len(lpb), len(mic), len(enh_sig))
                lpb, mic, enh_sig = lpb[:n], mic[:n], enh_sig[:n]
                talk_type = self._scenario_to_talk_type(scenario)
                scores = self._score_aecmos(talk_type, lpb, mic, enh_sig)
                erle = self._compute_erle(mic, enh_sig) if talk_type == "st" else None
            return self._send_json(200, {"scenario": scenario, "sample": sample, "enh": enh_src, "talk_type": talk_type, "erle": erle, **scores})
        except FileNotFoundError as e:
            return self._send_json(503, {"error": str(e), "hint": AECMOS_HINT})
        except Exception as e:
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    # ---- POST routes ----

    def do_POST(self):
        if self.path.startswith("/api/aecmos_upload"):
            return self._api_aecmos_upload()
        if self.path.startswith("/api/aecmos_score"):
            return self._api_aecmos_score()
        return self._send_json(404, {"error": "Unknown endpoint"})

    def _read_multipart(self, max_bytes: int = 100 * 1024 * 1024) -> dict[str, Any]:
        from evaluation.multipart import parse_content_type, parse_multipart
        ct = self.headers.get("content-type", "")
        ctype, params = parse_content_type(ct)
        if ctype != "multipart/form-data" or "boundary" not in params:
            raise ValueError("Expected multipart/form-data with boundary")
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 or content_length > max_bytes:
            raise ValueError("Invalid or too large Content-Length")
        body = io.BytesIO(self.rfile.read(content_length))
        return parse_multipart(body, params["boundary"])

    @staticmethod
    def _part_value(parts: dict, key: str, default: str = "") -> str:
        return parts[key].value.strip() if key in parts else default

    @staticmethod
    def _wav_from_bytes(data: bytes) -> np.ndarray:
        import librosa
        sig, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE)
        return sig

    def _api_aecmos_upload(self):
        try:
            parts = self._read_multipart()
            for key in ("lpb_wav", "mic_wav", "enh_wav"):
                if key not in parts:
                    return self._send_json(400, {"error": f"Missing required field: {key}"})

            talk_type = self._part_value(parts, "talk_type")
            scenario = self._part_value(parts, "scenario")
            if not talk_type and scenario:
                talk_type = self._scenario_to_talk_type(scenario)
            if talk_type not in {"nst", "st", "dt"}:
                return self._send_json(400, {"error": "talk_type must be nst, st, or dt"})

            lpb = self._wav_from_bytes(parts["lpb_wav"].data)
            mic = self._wav_from_bytes(parts["mic_wav"].data)
            enh = self._wav_from_bytes(parts["enh_wav"].data)
            n = min(len(lpb), len(mic), len(enh))
            lpb, mic, enh = lpb[:n], mic[:n], enh[:n]

            with _heavy_lock:
                scores = self._score_aecmos(talk_type, lpb, mic, enh)
                erle = self._compute_erle(mic, enh) if talk_type == "st" else None
            return self._send_json(200, {"talk_type": talk_type, "erle": erle, **scores})
        except ValueError as e:
            return self._send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            return self._send_json(503, {"error": str(e), "hint": AECMOS_HINT})
        except Exception as e:
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def _api_aecmos_score(self):
        """Score enhanced wav bytes posted alongside scenario/sample metadata.
        Used by the comparison page to get inline AECMOS for each method."""
        try:
            parts = self._read_multipart()
            if "enh_wav" not in parts:
                return self._send_json(400, {"error": "enh_wav file required"})

            scenario = self._part_value(parts, "scenario")
            sample = self._part_value(parts, "sample")
            method = self._part_value(parts, "method")
            if not scenario or not sample:
                return self._send_json(400, {"error": "scenario and sample required"})

            lpb, mic, missing = self._load_sample_pair(scenario, sample)
            if missing:
                return self._send_json(404, {"error": "Missing wav files", "missing": missing})

            enh = self._wav_from_bytes(parts["enh_wav"].data)

            n = min(len(lpb), len(mic), len(enh))
            lpb, mic, enh = lpb[:n], mic[:n], enh[:n]

            with _heavy_lock:
                talk_type = self._scenario_to_talk_type(scenario)
                scores = self._score_aecmos(talk_type, lpb, mic, enh)
                erle = self._compute_erle(mic, enh) if talk_type == "st" else None
            return self._send_json(200, {"method": method, "talk_type": talk_type, "erle": erle, **scores})
        except ValueError as e:
            return self._send_json(400, {"error": str(e)})
        except FileNotFoundError as e:
            return self._send_json(503, {"error": str(e), "hint": AECMOS_HINT})
        except Exception as e:
            print(f"[AECMOS score] 500: {e}", file=sys.stderr)
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def guess_type(self, path: str) -> str:
        if path.endswith(".wav"):
            return "audio/wav"
        if path.endswith(".json"):
            return "application/json"
        return super().guess_type(path)


def main() -> None:
    """Run the evaluation HTTP server; binds to PORT and serves comparison + AECMOS pages."""
    parser = argparse.ArgumentParser(description="AEC Evaluation server")
    parser.add_argument("--port", "-p", type=int, default=PORT, help=f"Port to bind (default: {PORT})")
    args = parser.parse_args()
    port = args.port

    project_root = _project_root()
    os.chdir(project_root)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if not _aecmos_model_path().exists():
        print(f"Note: AECMOS model not found — scoring will return 503 until you add {AECMOS_MODEL_NAME} to external/TSPNN/eval/ (see README).")
    print("Starting AEC Evaluation server...")
    print(f"Serving from: {project_root}")
    print(f"\n  Comparison: http://localhost:{port}/evaluation/index.html")
    print(f"  AECMOS:     http://localhost:{port}/evaluation/evaluate.html")
    print("\nPress Ctrl+C to stop.\n")
    with socketserver.ThreadingTCPServer(("", port), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
