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

## Size comparison

The original `NGGYU.mp4` is 75,234,064 bytes including audio. The video-only comparison target used for compression is 69,775,987 bytes. The quantized checkpoint contains both the frame generator and decoder and is 69,628,907 bytes.

| artifact | bytes | comparison |
| --- | ---: | ---: |
| original video, with audio | 75,234,064 | 100.00% |
| original video, video stream only | 69,775,987 | 100.00% baseline |
| quantized generator + decoder | 69,628,907 | 99.79% of video-only baseline |

Thus the quantized model is 147,080 bytes smaller than the video-only target. The separately rendered MP4 is not used for this comparison because it was encoded at a different video bitrate.

## Encoder-decoder training

The codec was trained as a single-video reconstruction model, starting from random initialization. All 5,306 source frames at 25 FPS were used; no temporal subsampling was used for codec training.

1. Each `1920 x 1080` frame was cropped to `(x=264, y=4, width=1392, height=1072)` to remove the side bars, then padded to `1408 x 1088` so that six 2x spatial reductions divide evenly.
2. The DCAE architecture used 32x spatial compression, 16 latent channels, and encoder/decoder widths `[64, 128, 256, 256, 512, 512]`.
3. The encoder and decoder were trained jointly in BF16. The reconstruction objective was `MSE + 0.1 * L1`, over every cropped source frame, for 200 epochs.
4. The resulting encoder produced BF16 latents with shape `[16, 34, 44]` for each frame. The encoder is not needed at inference time; the frame-index generator directly reconstructs these latents.
5. For the deployment decoder, a decoder-only QAT stage used the generated latents and the same frames. Decoder weights used symmetric groupwise INT4 fake quantization with group size 64 and BF16 compute for 30 epochs. The final checkpoint stores the decoder weights as actual packed INT4 nibbles with FP16 scales.

The frame-index generator is a rank-2560 PCA lookup: it stores one coefficient vector per frame and a shared basis, then computes `coeff[frame_index] @ basis` before denormalization and decoding. Its deployment artifact uses mixed symmetric INT4/INT8 groupwise quantization.
