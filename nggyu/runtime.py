from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image
import torch

from .modeling import Decoder

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS = {
    "int4": _ROOT / "checkpoints" / "nggyu_int4_int8.pt",
    "bf16": _ROOT / "checkpoints" / "nggyu_bf16.pt",
}


def _unpack_mixed(item: dict, device: torch.device) -> torch.Tensor:
    scale = item["scale"].to(device=device, dtype=torch.float32)
    axis = int(item["selection_axis"])
    mask = item["int8_selection"].to(device=device, dtype=torch.bool)
    n_groups = scale.shape[axis]
    q = torch.empty((*scale.shape, 64), device=device, dtype=torch.int16)
    packed = item["int4"].to(device=device)
    q4 = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int16) - 8
    q4 = q4.reshape(*scale.shape[:axis], int((~mask).sum()), *scale.shape[axis + 1:], 64)
    q.index_copy_(axis, (~mask).nonzero(as_tuple=False).flatten(), q4)
    q8 = item["int8"].to(device=device, dtype=torch.int16)
    q8 = q8.reshape(*scale.shape[:axis], int(mask.sum()), *scale.shape[axis + 1:], 64)
    q.index_copy_(axis, mask.nonzero(as_tuple=False).flatten(), q8)
    return (q.float() * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _unpack_decoder(item: dict, device: torch.device) -> torch.Tensor:
    packed = item["packed"].to(device=device)
    q = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1).to(torch.int16) - 8
    q = q[:, : int(item["cols"])]
    scale = item["scale"].to(device=device, dtype=torch.float32).repeat_interleave(int(item["group"]), dim=1)[:, : int(item["cols"])]
    shape = tuple(int(v) for v in item["shape"])
    moved = list(shape)
    axis = int(item["axis"])
    moved[0], moved[axis] = moved[axis], moved[0]
    return (q.float() * scale).reshape(moved).movedim(0, axis).to(torch.bfloat16)


def _make_decoder(state: dict, variant: str, device: torch.device) -> Decoder:
    model = Decoder().to(device=device, dtype=torch.bfloat16)
    decoded = {}
    for name, value in state.items():
        if variant == "int4" and isinstance(value, dict) and "packed" in value:
            decoded[name] = _unpack_decoder(value, device)
        elif isinstance(value, dict) and "value" in value:
            decoded[name] = value["value"].to(device=device)
        else:
            decoded[name] = value.to(device=device)
    model.load_state_dict(decoded, strict=True)
    return model.eval().requires_grad_(False)


@lru_cache(maxsize=4)
def _load(variant: str, device_name: str, checkpoint_name: str):
    if variant not in _DEFAULTS:
        raise ValueError("variant must be 'int4' or 'bf16'")
    device = torch.device(device_name)
    path = Path(checkpoint_name)
    package_path = _DEFAULTS[variant]
    if not path.is_absolute():
        path = package_path.parent / path
    ck = torch.load(path, map_location="cpu", weights_only=True)
    gen = ck["generator"]
    meta = gen["metadata"]
    if variant == "int4":
        coeff = _unpack_mixed(gen["coeff"], device)
        basis = _unpack_mixed(gen["basis"], device)
    else:
        coeff = gen["coeff"].to(device=device, dtype=torch.bfloat16)
        basis = gen["basis"].to(device=device, dtype=torch.bfloat16)
    decoder = _make_decoder(ck["decoder"]["model"], variant, device)
    return coeff, basis, decoder, meta


def NGGYU(frame_index: int, variant: str = "int4",
          device: Optional[Union[str, torch.device]] = None,
          checkpoint: Optional[Union[str, Path]] = None) -> Image.Image:
    """Generate one original-size RGB frame by zero-based frame index.

    ``NGGYU(4)`` returns a PIL RGB image for frame 4. The default variant is
    the packed INT4/INT8 model; use ``variant='bf16'`` for original precision.
    """
    if not isinstance(frame_index, (int, np.integer)):
        raise TypeError("frame_index must be an integer")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_name = str(checkpoint or _DEFAULTS[variant])
    coeff, basis, decoder, meta = _load(variant, str(device), checkpoint_name)
    index = int(frame_index)
    frames = int(meta["frames"])
    if index < 0 or index >= frames:
        raise IndexError(f"frame_index {index} outside [0, {frames})")
    with torch.inference_mode():
        z = (coeff[index:index + 1] @ basis).float()
        z = (z * float(meta["std"]) + float(meta["mean"])).to(torch.bfloat16)
        z = z.reshape(1, *[int(v) for v in meta["latent_shape"]])
        decoded = decoder(z).float().clamp(-1, 1)[0, :, :1072, :1392]
        crop = ((decoded + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[4:4 + 1072, 264:264 + 1392] = crop
    return Image.fromarray(frame, mode="RGB")
