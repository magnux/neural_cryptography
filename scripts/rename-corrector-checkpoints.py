#!/usr/bin/env python3
"""Update descriptive profile names in existing neural-corrector checkpoints.

With no paths, updates best.pt in these repository-relative run folders:
  checkpoints/neural-corrector/u-tube-burst5
  checkpoints/neural-corrector/u-tube-2x-burst5
  checkpoints/neural-corrector/u-tube-2x-attn-phase8

Only training.modelProfile changes. Weights, other metadata, format identifiers,
folder names and JSON reports are preserved. Run with the training environment
active, while no training process is writing these checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "neural-corrector"))

import torch
from checkpoint import MAX_CHECKPOINT_BYTES, load_checkpoint
from model import MODEL_PROFILES

RENAMES = {
    "normalized-tube": "U-tube",
    "longtube": "U-tube-2x",
    "attention-tube": "U-tube-2x-attn",
}
DEFAULT_PATHS = tuple(ROOT / "checkpoints" / "neural-corrector" / name / "best.pt"
                      for name in ("u-tube-burst5", "u-tube-2x-burst5", "u-tube-2x-attn-phase8"))


@dataclass
class Prepared:
    path: Path
    digest: str
    old: str | None
    new: str | None
    temporary: Path | None = None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(path: Path) -> Prepared:
    """Stage a validated current checkpoint without replacing the original."""
    path = Path(path).resolve(strict=True)
    info = path.stat()
    if not path.is_file() or not 0 < info.st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError(f"Invalid checkpoint file or size: {path}")
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError(f"Expanded checkpoint exceeds the current limit: {path}")
    original_digest = digest(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("training"), dict):
        raise ValueError(f"Missing checkpoint training metadata: {path}")
    old = checkpoint["training"].get("modelProfile")
    if old is not None and (not isinstance(old, str) or old not in (*RENAMES, *MODEL_PROFILES)):
        raise ValueError(f"Unknown checkpoint profile {old!r}: {path}")
    new = RENAMES.get(old, old)
    prepared = Prepared(path, original_digest, old, new)
    if old == new:
        # Missing optional metadata is valid when the current loader can infer it.
        load_checkpoint(path)
        return prepared
    checkpoint["training"]["modelProfile"] = new
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as target:
            prepared.temporary = Path(target.name)
            torch.save(checkpoint, target)
            target.flush()
            os.fsync(target.fileno())
        verified = load_checkpoint(prepared.temporary)
        # Also catch a serialization regression without changing the model schema.
        for name, value in checkpoint["modelState"].items():
            actual = verified["modelState"][name]
            if not torch.equal(value.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)):
                raise ValueError(f"Saved tensor changed: {name} in {path}")
        if verified["training"] != checkpoint["training"]:
            raise ValueError(f"Saved training metadata changed unexpectedly: {path}")
        os.chmod(prepared.temporary, stat.S_IMODE(info.st_mode))
        return prepared
    except BaseException:
        if prepared.temporary is not None:
            prepared.temporary.unlink(missing_ok=True)
        raise


def rename_checkpoints(paths, *, dry_run=False, emit=print) -> list[Prepared]:
    """Validate the whole selection first; replace each file atomically."""
    staged: list[Prepared] = []
    try:
        resolved = [Path(path).resolve(strict=True) for path in paths]
        if not resolved or len(resolved) != len(set(resolved)):
            raise ValueError("Select one or more distinct checkpoint files")
        for path in resolved:
            staged.append(prepare(path))
        # Do not overwrite a checkpoint that changed while being inspected.
        for item in staged:
            if digest(item.path) != item.digest:
                raise RuntimeError(f"Checkpoint changed during validation; retry after training stops: {item.path}")
        for item in staged:
            if item.temporary is None:
                emit(f"UNCHANGED  {item.path} ({item.new or 'profile inferred from weights'})")
            elif dry_run:
                emit(f"WOULD UPDATE  {item.path}: {item.old} -> {item.new}")
            else:
                if digest(item.path) != item.digest:
                    raise RuntimeError(f"Checkpoint changed before replacement: {item.path}")
                os.replace(item.temporary, item.path)
                emit(f"UPDATED  {item.path}: {item.old} -> {item.new}")
        return staged
    finally:
        for item in staged:
            if item.temporary is not None:
                item.temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="explicit .pt files (default: the three folders listed above)")
    parser.add_argument("--dry-run", action="store_true", help="validate and preview; leave original files unchanged")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        rename_checkpoints(args.paths or DEFAULT_PATHS, dry_run=args.dry_run)
    except Exception as error:
        print(f"Checkpoint metadata update failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
