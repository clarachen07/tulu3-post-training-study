"""Atomic BF16-bit chunks with immutable row ranges and complete SHA256 receipts."""

import json
import numpy as np
from .common import atomic_json, atomic_numpy, sha


def chunk_paths(folder, start):
    return folder / f"rows_{start:06d}.npy", folder / f"rows_{start:06d}.json"


def validate_chunk(folder, start, stop, signature, verify_bytes=True):
    path, receipt = chunk_paths(folder, start)
    if not receipt.exists():
        return None
    item = json.loads(receipt.read_text())
    if item["signature"] != signature or (item["start"], item["stop"]) != (start, stop):
        raise RuntimeError("Raw activation receipt identity mismatch")
    if item["filename"] != path.name or item["dtype"] != "uint16_bfloat16_bits":
        raise RuntimeError("Raw activation receipt format mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != item["shape"] or array.dtype != np.uint16:
        raise RuntimeError("Raw activation shape or dtype mismatch")
    if item["shape"] != [32, stop - start, 4096]:
        raise RuntimeError("Unexpected frozen raw dimensions")
    if verify_bytes and sha(path) != item["sha256"]:
        raise RuntimeError("Raw activation checksum mismatch")
    return item


def save_chunk(folder, start, values, signature):
    import torch

    if (
        values.dtype != torch.bfloat16
        or values.shape[0] != 32
        or values.shape[2] != 4096
    ):
        raise ValueError("Expected raw BF16 [32,batch,4096] block vectors")
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite activation chunk")
    stop = start + values.shape[1]
    path, receipt = chunk_paths(folder, start)
    atomic_numpy(path, values.contiguous().view(torch.uint16).numpy())
    item = {
        "signature": signature,
        "start": start,
        "stop": stop,
        "filename": path.name,
        "shape": list(values.shape),
        "dtype": "uint16_bfloat16_bits",
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }
    atomic_json(receipt, item)
    return item


def layer_tensor(folder, receipts, layer, count, device):
    import torch

    bits = np.empty((count, 4096), dtype=np.uint16)
    cursor = 0
    for item in receipts:
        if item["start"] != cursor:
            raise RuntimeError("Non-contiguous raw activation rows")
        chunk = np.load(folder / item["filename"], mmap_mode="r", allow_pickle=False)
        bits[item["start"] : item["stop"]] = chunk[layer]
        cursor = item["stop"]
    if cursor != count:
        raise RuntimeError("Incomplete raw activation rows")
    value = (
        torch.from_numpy(bits)
        .view(torch.bfloat16)
        .to(device=device, dtype=torch.float32)
    )
    if not torch.isfinite(value).all():
        raise RuntimeError("Nonfinite persisted activation")
    return value
