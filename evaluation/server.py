#!/usr/bin/env python3
"""
AEC Evaluation Server - Compare TSPNN (offline), DTLN-aec (online),
WebRTC AEC (online), and our model (online).

Serves the evaluation UI and provides APIs for enhancement and AECMOS scoring.
"""

import http.server
import socketserver
import os
from pathlib import Path
import json
import urllib.parse
import traceback
import io
import math
import wave
import glob
import sys
import tempfile
import subprocess

import numpy as np

PORT = 8888

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def _tspnn_root() -> Path:
    return _project_root() / "external" / "TSPNN"


def _results_root() -> Path:
    return _tspnn_root() / "results"


DTLN_DEFAULT_ROOT = (_project_root() / "external" / "DTLN-aec").resolve()
DTLN_DEFAULT_MODEL = (
    DTLN_DEFAULT_ROOT / "pretrained_models" / "dtln_aec_512" / "dtln_aec_512"
).resolve()

# ---------------------------------------------------------------------------
# Module-level caches (handler is instantiated per-request, so lru_cache on
# instance methods would never hit; keep caches here instead)
# ---------------------------------------------------------------------------

_ort_session_cache = None
_ckpt_model_cache: dict[str, tuple] = {}


def _get_ort_session():
    global _ort_session_cache
    if _ort_session_cache is not None:
        return _ort_session_cache
    import onnxruntime as ort
    model_path = _tspnn_root() / "eval" / "Run_1657188842_Stage_0.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"AECMOS model not found: {model_path}")
    _ort_session_cache = ort.InferenceSession(str(model_path))
    return _ort_session_cache


def _get_ckpt_model(ckpt_path_str: str):
    if ckpt_path_str in _ckpt_model_cache:
        return _ckpt_model_cache[ckpt_path_str]

    import torch
    tspnn = str(_tspnn_root())
    if tspnn not in sys.path:
        sys.path.insert(0, tspnn)
    from train.models import TSPNNBaseline

    ckpt = torch.load(ckpt_path_str, map_location="cpu")
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

    result = (model, cfg)
    if len(_ckpt_model_cache) >= 4:
        _ckpt_model_cache.pop(next(iter(_ckpt_model_cache)))
    _ckpt_model_cache[ckpt_path_str] = result
    return result


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _read_wav_16k(wav_path: Path) -> np.ndarray:
    import librosa
    sig, _ = librosa.load(str(wav_path), sr=16000)
    return sig


def _wav_bytes_pcm16(audio_f32: np.ndarray, sample_rate: int = 16000) -> bytes:
    x = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _corr(a: np.ndarray, b: np.ndarray):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if len(a) != len(b) or len(a) == 0:
        return None
    a, b = a - np.mean(a), b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _parse_qs(path: str) -> dict:
    parsed = urllib.parse.urlparse(path)
    return urllib.parse.parse_qs(parsed.query)


def _qval(q: dict, key: str, default=None):
    return (q.get(key) or [default])[0]


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

    def translate_path(self, path):
        path = urllib.parse.unquote(path).split("?", 1)[0].split("#", 1)[0]
        path = path.lstrip("/")
        root = _project_root()
        if path.startswith("results/"):
            return str(_results_root() / path[len("results/"):])
        if path.startswith("evaluation/"):
            return str(Path(__file__).parent / path[len("evaluation/"):])
        return str(root / path) if path else str(root)

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

    def _load_sample_pair(self, scenario: str, sample: str):
        blind = _results_root() / "blind_test_set_interspeech2021"
        lpb_path = blind / scenario / f"{sample}_lpb.wav"
        mic_path = blind / scenario / f"{sample}_mic.wav"
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

            sf.write(str(in_dir / f"{sample_name}_mic.wav"), np.asarray(mic_sig, dtype=np.float32), 16000)
            sf.write(str(in_dir / f"{sample_name}_lpb.wav"), np.asarray(lpb_sig, dtype=np.float32), 16000)

            python = os.environ.get("DTLN_PYTHON", sys.executable)
            proc = subprocess.run(
                [python, str(run_py), "-i", str(in_dir), "-o", str(out_dir), "-m", str(model_base)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"DTLN-aec failed: {proc.stderr}")

            out_path = out_dir / f"{sample_name}_mic.wav"
            if not out_path.exists():
                raise FileNotFoundError(f"DTLN output missing: {out_path}")
            import librosa
            enh, _ = librosa.load(str(out_path), sr=16000)
            return enh

    @staticmethod
    def _run_webrtc_aec(lpb_sig, mic_sig):
        import webrtc_audio_processing as wap

        sr = 16000
        frame_len = int(sr * 0.01)
        aec_type = int(os.environ.get("WEBRTC_AEC_TYPE", "3"))

        ap = wap.AudioProcessingModule(
            aec_type=aec_type,
            enable_ns=os.environ.get("WEBRTC_NS", "0") == "1",
            agc_type=1 if os.environ.get("WEBRTC_AGC", "0") == "1" else 0,
            enable_vad=os.environ.get("WEBRTC_VAD", "0") == "1",
        )
        ap.set_stream_format(sr, 1)
        ap.set_reverse_stream_format(sr, 1)

        n = min(len(lpb_sig), len(mic_sig))
        lpb_sig, mic_sig = lpb_sig[:n], mic_sig[:n]
        pad = (-n) % frame_len
        if pad:
            lpb_sig = np.pad(lpb_sig, (0, pad))
            mic_sig = np.pad(mic_sig, (0, pad))

        def to_i16(x):
            return (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype(np.int16).tobytes()

        out_i16 = np.zeros(len(mic_sig), dtype=np.int16)
        for i in range(0, len(mic_sig), frame_len):
            ap.process_reverse_stream(to_i16(lpb_sig[i:i + frame_len]))
            out_i16[i:i + frame_len] = np.frombuffer(
                ap.process_stream(to_i16(mic_sig[i:i + frame_len])), dtype=np.int16
            )

        out = out_i16.astype(np.float32) / 32768.0
        return out[:-pad] if pad else out

    @staticmethod
    def _enhance_with_ckpt(ckpt_path: Path, lpb_sig, mic_sig):
        import torch
        tspnn = str(_tspnn_root())
        if tspnn not in sys.path:
            sys.path.insert(0, tspnn)
        from train.utils.audio import StftConfig, stft, istft
        from train.utils.tdc import TdcConfig, tdc_align

        model, cfg = _get_ckpt_model(str(ckpt_path))
        sr = 16000
        n = min(len(lpb_sig), len(mic_sig))
        lpb_sig, mic_sig = lpb_sig[:n], mic_sig[:n]

        st = cfg.get("stft", {})
        stft_cfg = StftConfig(
            sample_rate=sr,
            win_ms=float(st.get("win_ms", 20)),
            hop_ms=float(st.get("hop_ms", 10)),
            n_fft=int(st.get("n_fft", 512)),
            center=bool(st.get("center", False)),
            alpha=float(st.get("alpha", 0.3)),
        )
        tc = cfg.get("tdc", {})
        tdc_cfg = TdcConfig(
            sample_rate=sr,
            max_delay_ms=float(tc.get("max_delay_ms", 200.0)),
            enabled=bool(tc.get("enabled", False)),
        )

        in_rms = float(np.sqrt(np.mean(np.square(mic_sig)) + 1e-8))
        norm = max(in_rms, 1e-8)

        with torch.no_grad():
            ref = torch.from_numpy(np.asarray(lpb_sig, np.float32)).unsqueeze(0) / norm
            mic = torch.from_numpy(np.asarray(mic_sig, np.float32)).unsqueeze(0) / norm
            ref = tdc_align(ref, mic, tdc_cfg)
            _, Ofine, _ = model(stft(ref, stft_cfg), stft(mic, stft_cfg))
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

    # ---- AECMOS scoring ----

    @staticmethod
    def _mel_transform(sample, sr):
        import librosa
        mel_spec = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=513, hop_length=256, n_mels=160)
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T

    @staticmethod
    def _compute_erle(echo_data, error_data):
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

        sr = 16000
        max_samples = 20 * sr
        if len(lpb_sig) >= max_samples:
            lpb_sig, mic_sig, enh_sig = lpb_sig[:max_samples], mic_sig[:max_samples], enh_sig[:max_samples]

        lpb_feat = self._mel_transform(lpb_sig, sr)
        mic_feat = self._mel_transform(mic_sig, sr)
        enh_feat = self._mel_transform(enh_sig, sr)

        ne_st = 1 if talk_type == "nst" else 0
        fe_st = 1 if talk_type == "st" else 0

        def pad_feat(feat, flag_val):
            return np.concatenate((feat, np.ones((20, feat.shape[1])) * flag_val, np.zeros((20, feat.shape[1]))), axis=0)

        mic_feat = pad_feat(mic_feat, 1 - fe_st)
        lpb_feat = pad_feat(lpb_feat, 1 - ne_st)
        enh_feat = pad_feat(enh_feat, 1)

        feats = np.expand_dims(np.stack((lpb_feat, mic_feat, enh_feat)).astype(np.float32), axis=0)
        sess = _get_ort_session()
        h0 = np.zeros((4, 1, 64), dtype=np.float32)
        result = sess.run([], {sess.get_inputs()[0].name: feats, "h0": h0})[0]
        return {"echo_mos": float(result[0]), "deg_mos": float(result[1])}

    @staticmethod
    def _scenario_to_talk_type(scenario: str):
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
        if not str(p).startswith(str(root) + os.sep):
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
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/evaluation/index.html")
            self.end_headers()
            return

        if self.path.startswith("/api/checkpoints"):
            return self._api_checkpoints()
        if self.path.startswith("/api/enhance_dtln_sample"):
            return self._api_enhance("dtln")
        if self.path.startswith("/api/enhance_webrtc_sample"):
            return self._api_enhance("webrtc")
        if self.path.startswith("/api/enhance_sample"):
            return self._api_enhance("ckpt")
        if self.path.startswith("/api/aecmos_sample"):
            return self._api_aecmos_sample()

        return super().do_GET()

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
                self.send_response(400)
                self.end_headers()
                return

            lpb, mic, missing = self._load_sample_pair(scenario, sample)
            if missing:
                return self._send_json(404, {"error": "Missing wav files", "missing": missing})

            if method == "dtln":
                enh = self._run_dtln(lpb, mic, sample)
            elif method == "webrtc":
                enh = self._run_webrtc_aec(lpb, mic)
            else:
                ckpt = _qval(q, "ckpt", "external/TSPNN/runs/full_aec_synth/ckpt_final.pt")
                max_len_s = _qval(q, "max_len_s")
                if max_len_s:
                    try:
                        n = int(float(max_len_s) * 16000)
                        if n > 0:
                            lpb, mic = lpb[:n], mic[:n]
                    except Exception:
                        pass
                ckpt_path = self._safe_ckpt_path(ckpt)
                enh = self._enhance_with_ckpt(ckpt_path, lpb, mic)

            if debug:
                extra = {"ckpt": str(ckpt_path)} if method == "ckpt" else {}
                return self._send_json(200, self._debug_stats(scenario, sample, lpb, mic, enh, **extra))

            return self._send_wav(enh)

        except ModuleNotFoundError as e:
            hints = {"dtln": "pip install tensorflow", "webrtc": "pip install ./external/python-webrtc-audio-processing", "ckpt": "pip install -r evaluation/requirements.txt"}
            return self._send_json(500, {"error": str(e), "hint": hints.get(method, "")})
        except Exception as e:
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

            enh_sig = _read_wav_16k(enh_path)
            n = min(len(lpb), len(mic), len(enh_sig))
            lpb, mic, enh_sig = lpb[:n], mic[:n], enh_sig[:n]

            talk_type = self._scenario_to_talk_type(scenario)
            scores = self._score_aecmos(talk_type, lpb, mic, enh_sig)
            erle = self._compute_erle(mic, enh_sig) if talk_type == "st" else None
            return self._send_json(200, {"scenario": scenario, "sample": sample, "enh": enh_src, "talk_type": talk_type, "erle": erle, **scores})
        except Exception as e:
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    # ---- POST routes ----

    def do_POST(self):
        if self.path.startswith("/api/aecmos_upload"):
            return self._api_aecmos_upload()
        return self._send_json(404, {"error": "Unknown endpoint"})

    def _api_aecmos_upload(self):
        try:
            import cgi
            import librosa

            ctype, _ = cgi.parse_header(self.headers.get("content-type", ""))
            if ctype != "multipart/form-data":
                return self._send_json(400, {"error": "Expected multipart/form-data"})

            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", ""), "CONTENT_LENGTH": self.headers.get("Content-Length", "0")},
            )

            talk_type = form.getfirst("talk_type", "").strip()
            scenario = form.getfirst("scenario", "").strip()
            if not talk_type and scenario:
                talk_type = self._scenario_to_talk_type(scenario)
            if talk_type not in {"nst", "st", "dt"}:
                return self._send_json(400, {"error": "talk_type must be nst, st, or dt"})

            def read_upload(field_name):
                if field_name not in form or not getattr(form[field_name], "file", None):
                    raise ValueError(f"Missing file: {field_name}")
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    tmp.write(form[field_name].file.read())
                    tmp.flush()
                    sig, _ = librosa.load(tmp.name, sr=16000)
                return sig

            lpb = read_upload("lpb_wav")
            mic = read_upload("mic_wav")
            enh = read_upload("enh_wav")
            n = min(len(lpb), len(mic), len(enh))
            lpb, mic, enh = lpb[:n], mic[:n], enh[:n]

            scores = self._score_aecmos(talk_type, lpb, mic, enh)
            erle = self._compute_erle(mic, enh) if talk_type == "st" else None
            return self._send_json(200, {"talk_type": talk_type, "erle": erle, **scores})
        except Exception as e:
            return self._send_json(500, {"error": str(e), "trace": traceback.format_exc()})

    def guess_type(self, path):
        if path.endswith(".wav"):
            return "audio/wav"
        if path.endswith(".json"):
            return "application/json"
        return super().guess_type(path)


def main():
    project_root = _project_root()
    os.chdir(project_root)
    print("Starting AEC Evaluation server...")
    print(f"Serving from: {project_root}")
    print(f"\n  Comparison: http://localhost:{PORT}/evaluation/index.html")
    print(f"  AECMOS:     http://localhost:{PORT}/evaluation/evaluate.html")
    print("\nPress Ctrl+C to stop.\n")
    with socketserver.ThreadingTCPServer(("", PORT), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
