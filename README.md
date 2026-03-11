# offline-aec

AEC comparison: TSPNN (offline), DTLN-aec (online), WebRTC (online), and your model (online).

```
offline-aec/
├── external/          # Submodules: TSPNN, DTLN-aec, python-webrtc-audio-processing
├── model/             # Your model + training (checkpoints in runs/)
├── evaluation/        # Comparison UI + AECMOS server
└── README.md
```

## Setup

### 1. Clone and init submodules

```bash
git clone --recursive <this-repo-url>
cd offline-aec
```

If already cloned:

```bash
git submodule update --init --recursive
```

### 2. Install deps

```bash
pip install -r evaluation/requirements.txt

# Optional baselines:
# pip install ./external/python-webrtc-audio-processing   # WebRTC (needs SWIG)
# pip install tensorflow                                  # DTLN-aec
```

### 3. Run server

```bash
python evaluation/server.py
```

- Comparison: http://localhost:8888/evaluation/index.html  
- AECMOS: http://localhost:8888/evaluation/evaluate.html

## Comparison Methods

| Method | Mode | Source |
|--------|------|--------|
| **TSPNN** | Offline | Precomputed from `external/TSPNN/results/` |
| **DTLN-aec** | Online | Subprocess via `external/DTLN-aec/run_aec.py` |
| **WebRTC AEC** | Online | `external/python-webrtc-audio-processing` |
| **Your model** | Online | Checkpoints in `model/runs/` or `external/TSPNN/runs/` |

## Submodule Details

- **TSPNN**: https://github.com/enhancer12/TSPNN — Two-stage progressive neural network (INTERSPEECH 2023)
- **DTLN-aec**: https://github.com/breizhn/DTLN-aec — Real-time AEC (AEC-Challenge 2021, 3rd place)
- **python-webrtc-audio-processing**: https://github.com/xiongyihui/python-webrtc-audio-processing — WebRTC APM Python bindings
