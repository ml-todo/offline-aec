# offline-aec

Compare acoustic echo cancellation (AEC) methods: **TSPNN** (offline), **DTLN-aec**, **WebRTC AEC3**, and your own model. Run the evaluation server and open the comparison UI in your browser.

**Prerequisites:** Python 3.8+ (and optionally Node.js + npm for the WebRTC AEC3 track).

---

## Quick start (minimal setup)

**1. Clone and init submodules**

```bash
git clone --recursive <repository-url>
cd offline-aec
```

If already cloned:

```bash
git submodule update --init --recursive
```

**2. Install Python dependencies**

```bash
pip install -r evaluation/requirements.txt
```

**3. Start the server**

```bash
python evaluation/server.py
```

**4. Open the comparison page**

- **Comparison UI:** http://localhost:8888/evaluation/index.html  
- **AECMOS (scores):** http://localhost:8888/evaluation/evaluate.html  

Default port is 8888. Use `--port` to change it: `python evaluation/server.py --port 8889`.

---

## Data needed for the comparison UI

The comparison page loads audio from:

- **Input (far-end, near-end):** `external/TSPNN/results/blind_test_set_interspeech2021/<scenario>/<sample>_lpb.wav` and `<sample>_mic.wav`
- **Sample list:** `evaluation/samples_*.json` (included in the repo; point to sample IDs)

If you have the [AEC Challenge](https://github.com/microsoft/AEC-Challenge) blind test set or your own data, place WAVs in that layout under `external/TSPNN/results/`. If you have TSPNN output, run:

```bash
python evaluation/generate_samples.py
```

to refresh the sample lists from `external/TSPNN/results/output/ours/`. Without data, the UI will show “file not found” for tracks; the server and pages still run.

---

## Repository layout

```
offline-aec/
├── evaluation/           # Comparison UI and server
│   ├── server.py        # HTTP server + APIs
│   ├── index.html       # Main comparison page
│   ├── evaluate.html    # AECMOS scoring page
│   ├── requirements.txt
│   ├── webrtc-aec3-node/   # Optional: WebRTC AEC3 (Node)
│   └── README.md        # Evaluation setup details
├── external/            # Submodules
│   ├── TSPNN/           # TSPNN model + results layout
│   └── DTLN-aec/        # DTLN-aec baseline
├── model/               # Your model and training
└── README.md            # This file
```

---

## Optional features

### WebRTC AEC3 track

To show the “WebRTC AEC3” track in the comparison UI:

1. Install [Node.js](https://nodejs.org/) (includes npm).
2. In the repo:

   ```bash
   cd evaluation/webrtc-aec3-node && npm install && cd ../..
   ```

If Node/npm or the runner is missing, that track shows an error; other tracks still work.

### AECMOS scores (echo / degradation MOS)

To show echo and degradation scores in the UI, add the AECMOS ONNX model:

1. Download from [Microsoft AEC-Challenge (AECMOS)](https://github.com/microsoft/AEC-Challenge/tree/main/AECMOS).
2. Place `Run_1657188842_Stage_0.onnx` in `external/TSPNN/eval/`.

Without it, AECMOS requests return 503; the comparison UI and other features still work.

### DTLN-aec track

Requires the DTLN-aec submodule and `pip install tensorflow` (or use the env vars below to point to another Python). The server uses `external/DTLN-aec/run_aec.py` by default.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DTLN_ROOT` | `external/DTLN-aec` | DTLN-aec repo path |
| `DTLN_MODEL` | (auto) | DTLN model dir; auto if unset |
| `DTLN_PYTHON` | `sys.executable` | Python for DTLN subprocess |
| `DTLN_TIMEOUT` | `120` | DTLN subprocess timeout (s) |
| `WEBRTC_AEC3_NODE` | `node` | Node binary for AEC3 |
| `WEBRTC_AEC3_TIMEOUT` | `60` | AEC3 subprocess timeout (s) |
| `CKPT_PYTHON` | `sys.executable` | Python for checkpoint enhancement |
| `CKPT_ENHANCE_TIMEOUT` | `120` | Checkpoint subprocess timeout (s) |
| `CKPT_MAX_LEN_S` | `30` | Max input length (s) for ckpt; `0` = no trim |

---

## Comparison methods

| Method | Mode | Source |
|--------|------|--------|
| **TSPNN** | Offline | `external/TSPNN/results/output/ours/` |
| **DTLN-aec** | Subprocess | `external/DTLN-aec/run_aec.py` |
| **WebRTC AEC3** | Subprocess (Node) | `evaluation/webrtc-aec3-node` |
| **Your model** | Subprocess | Checkpoints in `model/runs/` or `external/TSPNN/runs/` |

---

## Documentation

- **[evaluation/README.md](evaluation/README.md)** — Evaluation server and UI setup, sample lists, optional features.
- **[evaluation/webrtc-aec3-node/README.md](evaluation/webrtc-aec3-node/README.md)** — WebRTC AEC3 Node runner setup.

---

## References

- **TSPNN:** [enhancer12/TSPNN](https://github.com/enhancer12/TSPNN) — Two-stage progressive neural network (INTERSPEECH 2023).
- **DTLN-aec:** [breizhn/DTLN-aec](https://github.com/breizhn/DTLN-aec) — Real-time AEC (AEC Challenge 2021).
- **AEC Challenge:** [microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge).
