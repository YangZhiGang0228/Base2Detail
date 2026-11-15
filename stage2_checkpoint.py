
import json
import os
import struct

import numpy as np
import torch


STAGE2_QUANTIZED_CHECKPOINT = "netparams_disp_10bit.dat"
QUANTIZED_FORMAT = "al-q1-anchor-distance-disp-v1"


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
    limit = (1 << (bits - 1)) - 1
    codes = []
    tensors = []
    for name, tensor in model.net_disp.state_dict().items():
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        scale = max(float(np.max(np.abs(array))) / limit,
                    float(np.finfo(np.float32).tiny))
        codes.append(
            np.clip(np.rint(array / scale), -limit, limit).astype(np.int16).reshape(-1))
        tensors.append({
            "name": name,
            "shape": list(array.shape),
            "count": int(array.size),
            "scale": scale,
        })
    payload = _pack_signed_codes(np.concatenate(codes), bits)
    header = json.dumps({
        "format": QUANTIZED_FORMAT,
        "bits": int(bits),
        "output_scale": float(model.output_scale),
        "tensors": tensors,
        "payload_bytes": len(payload),
    }, separators=(",", ":")).encode("utf-8")
    temporary = str(path) + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(struct.pack("<I", len(header)))
        stream.write(header)
        stream.write(payload)
    os.replace(temporary, path)


def load_quantized_disp(model, path, device="cpu"):
    with open(path, "rb") as stream:
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size).decode("utf-8"))
        if header["format"] != QUANTIZED_FORMAT:
            raise ValueError("unsupported Stage-2 checkpoint format")
        payload = stream.read(int(header["payload_bytes"]))

    tensor_headers = header["tensors"]
    values = _unpack_signed_codes(
        payload, sum(int(item["count"]) for item in tensor_headers),
        int(header["bits"]))
    state = {}
    offset = 0
    for item in tensor_headers:
        count = int(item["count"])
        array = values[offset:offset + count].astype(np.float32)
        array *= np.float32(item["scale"])
        state[item["name"]] = torch.from_numpy(
            array.reshape(tuple(int(value) for value in item["shape"])))
        offset += count
    model.net_disp.load_state_dict(state, strict=True)
    model.output_scale = float(header["output_scale"])
    saved_anchors = header.get("anchor_idx")
    if saved_anchors is not None:
        model.anchor_idx.copy_(torch.tensor(
            saved_anchors, dtype=torch.long, device=model.anchor_idx.device))
    model.anchor_ready.fill_(saved_anchors is not None)
    return model.to(device)
