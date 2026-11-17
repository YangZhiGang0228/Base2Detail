import os
import struct

import numpy as np
import torch


BASE_CHECKPOINT = "netparams_base.dat"
BASE_MAGIC = b"ALB1"
NORMALIZATION = struct.Struct("<4f")


def save_base_checkpoint(model, normalization, path):
    state = model.state_dict()
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("The compact Stage 1 format requires FP32 parameters.")
    center = torch.as_tensor(normalization["center"]).tolist()
    scale = float(normalization["scale"])
    temporary = str(path) + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(BASE_MAGIC)
        stream.write(NORMALIZATION.pack(*center, scale))
        for tensor in state.values():
            values = tensor.detach().cpu().numpy().astype("<f4", copy=False)
            stream.write(values.tobytes())
    os.replace(temporary, path)


def load_base_checkpoint(path, *, model=None, require_normalization=False):
    with open(path, "rb") as stream:
        compact = stream.read(len(BASE_MAGIC)) == BASE_MAGIC
        if compact:
            if model is None:
                raise ValueError("The compact Stage 1 checkpoint requires a model configuration.")
            x, y, z, scale = NORMALIZATION.unpack(stream.read(NORMALIZATION.size))
            normalization = {
                "center": torch.tensor([x, y, z], dtype=torch.float32),
                "scale": torch.tensor(scale, dtype=torch.float32),
            }
            template = model.state_dict()
            payload = stream.read()
            if len(payload) != 4 * sum(tensor.numel() for tensor in template.values()):
                raise ValueError("Stage 1 checkpoint size does not match the model configuration.")
            values = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
            state = {}
            offset = 0
            for name, tensor in template.items():
                end = offset + tensor.numel()
                state[name] = torch.from_numpy(values[offset:end]).reshape(tensor.shape)
                offset = end
    if not compact:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        normalization = checkpoint.get("mesh_normalization")
        state = checkpoint if normalization is None else checkpoint["state_dict"]
    if normalization is None:
        if require_normalization:
            raise ValueError(
                "This Stage 1 checkpoint has no mesh normalization. "
                "Use a checkpoint saved with the training normalization.")
        return state, None

    center = normalization["center"]
    scale = normalization["scale"]
    if (center.shape != (3,) or scale.ndim != 0 or not torch.isfinite(center).all()
            or not torch.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid mesh normalization in the Stage 1 checkpoint.")
    return state, normalization
