# Evaluation server and comparison UI

This directory contains the AEC comparison web UI and the HTTP server that backs it. New users can get the evaluation page running in a few steps from the repo root (see [../README.md](../README.md)); this file adds evaluation-specific details.

---

## Quick start (from repo root)

```bash
pip install -r evaluation/requirements.txt
python evaluation/server.py
```

Then open **http://localhost:8888/evaluation/index.html**.

---

## What’s included

| File / directory      | Role |
|-----------------------|------|
| `server.py`           | HTTP server: serves HTML, static files, and APIs for enhancement and AECMOS. |
| `index.html`          | Main comparison page: input/output waveforms, play/pause, synced cursors, optional AECMOS. |
| `evaluate.html`       | AECMOS-only page for uploading WAVs and getting scores. |
| `requirements.txt`    | Python deps (librosa, numpy, onnxruntime, soundfile, torch). |
| `run_ckpt_enhance.py` | Subprocess entrypoint for “Our model” enhancement. |
| `generate_samples.py` | Builds `samples_*.json` from TSPNN output under `external/TSPNN/results/output/ours/`. |
| `multipart.py`        | Multipart form parsing for AECMOS upload/score. |
| `webrtc-aec3-node/`   | Optional Node runner for WebRTC AEC3; run `npm install` there if you want that track. |

---

## Sample lists

The comparison UI reads sample IDs from:

- `evaluation/samples_doubletalk.json`
- `evaluation/samples_farend_singletalk.json`
- `evaluation/samples_nearend_singletalk.json`

These are committed with default content. To regenerate them from TSPNN output:

```bash
python evaluation/generate_samples.py
```

Requires `external/TSPNN/results/output/ours/<scenario>/` to contain `*_mic.wav` files.

---

## Optional: WebRTC AEC3

To enable the “WebRTC AEC3” track:

```bash
cd evaluation/webrtc-aec3-node && npm install && cd ../..
```

See [webrtc-aec3-node/README.md](webrtc-aec3-node/README.md) for details.

---

## Optional: AECMOS scores

Place the AECMOS ONNX model at:

`external/TSPNN/eval/Run_1657188842_Stage_0.onnx`

Download from [Microsoft AEC-Challenge (AECMOS)](https://github.com/microsoft/AEC-Challenge/tree/main/AECMOS). Without it, AECMOS endpoints return 503; the comparison UI still works.

---

## APIs used by the UI

- `GET /api/checkpoints` — List checkpoint paths for “Our model”.
- `GET /api/out_sample?scenario=&sample=` — TSPNN output WAV (level-matched).
- `GET /api/enhance_dtln_sample?scenario=&sample=` — DTLN-aec output WAV.
- `GET /api/enhance_webrtc_aec3_sample?scenario=&sample=` — WebRTC AEC3 output WAV.
- `GET /api/enhance_sample?scenario=&sample=&ckpt=` — Checkpoint enhancement WAV.
- `POST /api/aecmos_score` — Score an enhanced WAV (multipart: scenario, sample, method, enh_wav).

