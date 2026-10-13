"""Content identities for the local weight files selected by Transformers.

Hashing streams checkpoint bytes without importing Torch or deserializing pickle
weights. The aggregate SHA256 identifies a canonical manifest of file names,
sizes and content SHA256 values; it is not a stat/mtime fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def local_checkpoint_identity(model_dir: str | Path) -> dict[str, Any]:
    """Hash the exact default local weight selection used by ``from_pretrained``.

    Safetensors takes precedence over PyTorch bin files, and a monolithic file
    takes precedence over its index. For sharded checkpoints both the index and
    every referenced shard are included. Unreferenced files are not loaded by
    the framework and do not affect this identity.
    """
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("model_dir must be a local pretrained model directory")
    selection = next(
        (
            name
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            if (root / name).is_file()
        ),
        None,
    )
    if selection is None:
        raise ValueError(f"no supported local checkpoint weights found under {root}")
    names = [selection]
    index_sha256 = None
    if selection.endswith(".index.json"):
        index_bytes = (root / selection).read_bytes()
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        index = json.loads(index_bytes)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid checkpoint weight_map: {root / selection}")
        suffix = ".safetensors" if selection.startswith("model.safetensors") else ".bin"
        for name in weight_map.values():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not name.endswith(suffix)
            ):
                raise ValueError(f"invalid checkpoint shard path: {name!r}")
        names.extend(sorted(set(weight_map.values())))
    files = []
    markers = []
    for name in names:
        path = root / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(8 * 1024**2):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        marker = _stat_marker(before)
        if marker != _stat_marker(after) or marker != _stat_marker(path.stat()):
            raise ValueError(f"checkpoint changed while hashing: {path}")
        files.append(
            {"name": name, "size_bytes": before.st_size, "sha256": digest.hexdigest()}
        )
        markers.append((path, marker))
    for path, marker in markers:
        if marker != _stat_marker(path.stat()):
            raise ValueError(f"checkpoint changed while hashing: {path}")
    if index_sha256 is not None and files[0]["sha256"] != index_sha256:
        raise ValueError("checkpoint index changed while resolving its shards")
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256_content_manifest_v1",
        "model_dir": str(root),
        "selection": selection,
        "format": "safetensors"
        if selection.startswith("model.safetensors")
        else "pytorch_bin",
        "sha256": hashlib.sha256(
            b"fakegpu.local_checkpoint.v1\n" + manifest
        ).hexdigest(),
        "total_bytes_read": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def bind_local_checkpoint(
    training_config: Mapping[str, Any], model_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a copied configuration to the actual resolved checkpoint directory.

    A caller-supplied digest is an expected identity, not a replacement for
    reading the weights. This rejects stale identities after an in-place edit.
    """
    identity = local_checkpoint_identity(model_dir)
    expected = training_config.get("checkpoint_sha256")
    if expected is not None and str(expected).lower() != identity["sha256"]:
        raise ValueError(
            "checkpoint_sha256 does not match the local checkpoint content"
        )
    config = dict(training_config)
    config["model_name_or_path"] = identity["model_dir"]
    config["checkpoint_sha256"] = identity["sha256"]
    return config, identity


def _stat_marker(stat: os.stat_result) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
