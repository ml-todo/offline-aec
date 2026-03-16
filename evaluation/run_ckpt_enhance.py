#!/usr/bin/env python3
"""
Run checkpoint enhancement in a subprocess.
Used by the evaluation server so OOM or crashes in the model don't kill the server.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AEC enhancement with a checkpoint")
    parser.add_argument("--lpb", required=True, help="Path to far-end reference WAV (16 kHz)")
    parser.add_argument("--mic", required=True, help="Path to microphone WAV (16 kHz)")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out", required=True, help="Path to write enhanced WAV")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import librosa
    import soundfile as sf
    from model.enhance import SAMPLE_RATE, enhance

    lpb, _ = librosa.load(args.lpb, sr=SAMPLE_RATE)
    mic, _ = librosa.load(args.mic, sr=SAMPLE_RATE)
    out_sig = enhance(lpb, mic, args.ckpt)
    sf.write(args.out, out_sig.astype("float32"), SAMPLE_RATE)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as e:
        import traceback
        print(f"run_ckpt_enhance failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
