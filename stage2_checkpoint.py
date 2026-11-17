
import os
import struct

import numpy as np
import torch


STAGE2_QUANTIZED_CHECKPOINT = "netparams_disp_10bit.dat"
DISP_MAGIC = b"ALD2"
DISP_METADATA = struct.Struct("<Bd")


def _pack_signed_codes(values, bits=10):
    mask = (1 << bits) - 1
    accumulator = 0
    available = 0
    payload = bytearray()
    for value in np.asarray(values, dtype=np.int64).reshape(-1):
        accumulator |= (int(value) & mask) << available
        available += bits
        while available >= 8:
            payload.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        payload.append(accumulator & 0xFF)
    return bytes(payload)


def _unpack_signed_codes(payload, count, bits=10):
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    accumulator = 0
    available = 0
    offset = 0
    result = np.empty(int(count), dtype=np.int16)
    for index in range(int(count)):
        while available < bits:
            accumulator |= payload[offset] << available
            offset += 1
            available += 8
        value = accumulator & mask
        accumulator >>= bits
        available -= bits
        result[index] = value - (1 << bits) if value & sign else value
    return result


def save_quantized_disp(model, path, bits=10):
    if not model.anchor_ready.item():
        raise ValueError("Select the Stage 2 anchors before saving the checkpoint.")
    anchors = model.anchor_idx.detach().cpu().tolist()
    limit = (1 << (bits - 1)) - 1
    codes = []
    scales = []
    for tensor in model.net_disp.state_dict().values():
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        scale = max(float(np.max(np.abs(array))) / limit,
                    float(np.finfo(np.float32).tiny))
        codes.append(
            np.clip(np.rint(array / scale), -limit, limit).astype(np.int16).reshape(-1))
        scales.append(scale)
    payload = _pack_signed_codes(np.concatenate(codes), bits)
    _write_compact_disp(path, bits, model.output_scale, scales, payload, anchors)


def _write_compact_disp(path, bits, output_scale, scales, payload, anchors):
    temporary = str(path) + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(DISP_MAGIC)
        stream.write(DISP_METADATA.pack(bits, output_scale))
        stream.write(struct.pack(f"<{len(scales)}f", *scales))
        stream.write(struct.pack(f"<{len(anchors)}I", *anchors))
        stream.write(payload)
    os.replace(temporary, path)


def load_quantized_disp(model, path, device="cpu"):
    with open(path, "rb") as stream:
        if stream.read(len(DISP_MAGIC)) != DISP_MAGIC:
            raise ValueError("Stage 2 decoding requires a checkpoint with stored anchor indices (ALD2).")
        template = model.net_disp.state_dict()
        scale_format = struct.Struct(f"<{len(template)}f")
        anchor_format = struct.Struct(f"<{model.anchor_idx.numel()}I")
        bits, output_scale = DISP_METADATA.unpack(stream.read(DISP_METADATA.size))
        scales = scale_format.unpack(stream.read(scale_format.size))
        anchors = anchor_format.unpack(stream.read(anchor_format.size))
        payload = stream.read()

    count = sum(tensor.numel() for tensor in template.values())
    if not 2 <= bits <= 16 or len(payload) != (count * bits + 7) // 8:
        raise ValueError("Stage 2 checkpoint size or bit depth does not match the model configuration.")
    values = _unpack_signed_codes(payload, count, bits)
    state = {}
    offset = 0
    for (name, tensor), scale in zip(template.items(), scales):
        end = offset + tensor.numel()
        array = values[offset:end].astype(np.float32)
        array *= np.float32(scale)
        state[name] = torch.from_numpy(array).reshape(tensor.shape)
        offset = end
    model.net_disp.load_state_dict(state, strict=True)
    model.output_scale = output_scale
    model.anchor_idx.copy_(model.anchor_idx.new_tensor(anchors))
    model.anchor_ready.fill_(True)
    return model.to(device)
