# Our AEC Model

This directory is for **your model** and training code. Place trained checkpoints in `runs/`; they will appear in the evaluation page's checkpoint dropdown.

The evaluation server loads checkpoints using the same interface as TSPNN (see `external/TSPNN/train/models/tspnn.py`). To add your own model:

- Reuse the TSPNN training scaffold: copy from `external/TSPNN/train/` and adapt
- Or implement a compatible checkpoint format (PyTorch state dict + `cfg` dict)

See `external/TSPNN/train/README.md` for training setup and data preparation.
