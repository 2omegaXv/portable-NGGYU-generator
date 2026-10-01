# nggyu

`NGGYU(frame_index)` returns one reconstructed RGB frame as a PIL image. The index is zero based and the returned image keeps the original `1920 x 1080` canvas, including the black side bars.

```python
from nggyu import NGGYU

img = NGGYU(4)                         # packed INT4/INT8 model
img.save("frame_0004.png")
img = NGGYU(4, variant="bf16")         # original BF16 model
```

Install from this directory with `pip install .`. The package requires PyTorch, Diffusers `0.32.2`, and Pillow. CUDA is used automatically when available; pass `device="cpu"` or `device="cuda"` to override it.

The two included checkpoints are the only model artifacts:

- `checkpoints/nggyu_int4_int8.pt`: mixed INT4/INT8 frame generator plus packed INT4 decoder.
- `checkpoints/nggyu_bf16.pt`: original BF16 frame generator plus BF16 decoder.

The first call loads the selected checkpoint and later calls reuse it in the process. Frames are generated from the integer frame index; no source video or latent cache is required at inference time.
