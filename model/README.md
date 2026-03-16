# Our AEC Model

This directory provides the enhancement interface used by the evaluation server.

## How it works

The server calls `model.enhance.enhance(lpb, mic, checkpoint_path)` for the "Our Model" track. The default implementation loads TSPNN-compatible checkpoints from `external/TSPNN/train/`.

## Adding your own model

Edit `enhance.py` and replace `_load_model()` with your architecture. The contract is:

```python
def enhance(lpb: np.ndarray, mic: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """
    Args:
        lpb: far-end reference, float32, 16 kHz mono
        mic: microphone signal, float32, 16 kHz mono
        checkpoint_path: path to .pt checkpoint
    Returns:
        enhanced signal, float32, 16 kHz mono
    """
```

Place checkpoints in `runs/`; they appear in the evaluation page dropdown.
