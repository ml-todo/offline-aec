#!/usr/bin/env python3
"""Generate sample list JSON from external/TSPNN/results. Run: python evaluation/generate_samples.py"""
from __future__ import annotations

import json
from pathlib import Path


def get_sample_base_names(directory: Path) -> list[str]:
    """List base names (stem minus '_mic') of *_mic.wav files in directory."""
    if not directory.exists():
        print(f"Warning: {directory} not found")
        return []
    return sorted({f.stem[:-4] for f in directory.iterdir() if f.name.endswith("_mic.wav")})


def main() -> None:
    root = Path(__file__).parent.parent
    output_dir = root / "external" / "TSPNN" / "results" / "output" / "ours"
    eval_dir = Path(__file__).parent

    for name, folder in [("doubletalk", "doubletalk"), ("farend_singletalk", "farend-singletalk"), ("nearend_singletalk", "nearend-singletalk")]:
        samples = get_sample_base_names(output_dir / folder)
        out = eval_dir / f"samples_{name}.json"
        out.write_text(json.dumps(samples, indent=2))
        print(f"{out.name}: {len(samples)} samples")


if __name__ == "__main__":
    main()
