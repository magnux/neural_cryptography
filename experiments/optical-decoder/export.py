#!/usr/bin/env python3
"""Export the optical decoder and deterministic parity fixtures.

Without a checkpoint, the weights are untrained public diagnostic material.
With a checkpoint, export the trained weights and public training provenance.
The numerical benchmark verifies runtime agreement, not recovery accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import torch

from contract import (ATOL, CHANNELS, CLASSES, FORMAT, INPUT_NAME, OUTPUT_NAME,
                      PARAMETER_COUNT, PIXELS_PER_CELL, RTOL, SIZES, VERSION,
                      describe_file, fixture_input, shapes, write_f32)
from model import OpticalDecoder, parameter_count
from checkpoint import load_checkpoint, training_provenance


OPSET = 17
ALLOWED_OPERATORS = {"Conv", "Relu", "Concat", "Resize", "Constant"}


def native_parity(model_path: Path, pairs: Path, threads: int) -> list[dict]:
    """Run native inference out of process: a native backend failure must be explicit."""
    script = """
import json, sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, sys.argv[4])
from contract import parity_metrics
options = ort.SessionOptions()
options.intra_op_num_threads = int(sys.argv[3])
options.inter_op_num_threads = 1
session = ort.InferenceSession(sys.argv[1], sess_options=options, providers=['CPUExecutionProvider'])
checks = []
with np.load(sys.argv[2], allow_pickle=False) as data:
    for i in range(2):
        actual = session.run(['logits'], {'pixels': data[f'input{i}']})[0]
        checks.append({'fixtureSeed': int(data[f'seed{i}']), **parity_metrics(data[f'expected{i}'], actual)})
print(json.dumps(checks))
"""
    try:
        result = subprocess.run([sys.executable, "-c", script, str(model_path), str(pairs),
                                 str(threads), str(Path(__file__).resolve().parent)],
                                capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Python ONNX Runtime check exceeded 120 seconds") from error
    if result.returncode:
        detail = result.stderr.strip()[-1500:]
        raise RuntimeError(f"Python ONNX Runtime failed (exit {result.returncode}): {detail}\n"
                           "Use --skip-ort-check only to explicitly export an unchecked bundle for browser parity testing.")
    checks = json.loads(result.stdout)
    if len(checks) != 2 or not all(check["passed"] for check in checks):
        raise RuntimeError(f"Python ONNX Runtime parity failed: {checks}")
    return checks


def export_smoke(out: Path, seed: int = 7, sizes: tuple[int, ...] = SIZES, threads: int = 1,
                 check_ort: bool = True, checkpoint: Path | None = None) -> dict:
    """Export one initialized or checkpoint-loaded set of weights on CPU.

    Validate every graph and, unless explicitly disabled, two independent inputs
    per size against Python ONNX Runtime before publishing manifest.json.
    The CLI always exports both browser profiles; subsets are for unit tests.
    Existing unrelated output files are not deleted.
    """
    if not sizes or len(sizes) != len(set(sizes)):
        raise ValueError("Choose each requested grid size once")
    for size in sizes:
        shapes(size)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 64:
        raise ValueError("Threads must be between 1 and 64")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = OpticalDecoder().cpu().eval()
    provenance = {}
    if checkpoint is not None:
        loaded = load_checkpoint(checkpoint, model)
        provenance = {"checkpointSha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                      "training": training_provenance(loaded["training"])}
    count = parameter_count(model)
    if count != PARAMETER_COUNT:
        raise RuntimeError(f"Model contract changed: expected {PARAMETER_COUNT} parameters, got {count}")
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("torch", "numpy", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    manifest = {
        "format": FORMAT, "version": VERSION, "status": "trained" if checkpoint is not None else "untrained", "seed": seed,
        "parameterCount": count, "pixelsPerCell": PIXELS_PER_CELL, "channels": CHANNELS,
        "classes": CLASSES, "inputName": INPUT_NAME, "outputName": OUTPUT_NAME, "opset": OPSET,
        "versions": versions,
        "dtype": "float32", "byteOrder": "little", "layout": "NCHW",
        "normalization": {"divisor": 127.5, "offset": -1.0, "clip": [-1.5, 1.5]},
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "pythonParityStatus": {"status": "passed" if check_ort else "not_run",
                               "reason": None if check_ort else "Explicit --skip-ort-check; Python ONNX Runtime parity not run."},
        "cases": [],
        **provenance,
    }
    with tempfile.TemporaryDirectory(prefix=".optical-export-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for size in sizes:
            input_shape, output_shape = shapes(size)
            pixels = fixture_input(size, seed)
            with torch.inference_mode():
                logits = model(torch.from_numpy(pixels)).numpy()
            if logits.shape != output_shape:
                raise RuntimeError(f"Unexpected output shape: {logits.shape}")
            model_path = stage / f"model-{size}.onnx"
            # Explicit legacy exporter avoids requiring onnxscript and emits a
            # small Conv/Relu/Resize/Concat graph with static batch-one shapes.
            torch.onnx.export(model, (torch.from_numpy(pixels),), str(model_path),
                              input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                              opset_version=OPSET, dynamo=False, do_constant_folding=True)
            graph = onnx.load(model_path)
            onnx.checker.check_model(graph, full_check=True)
            operators = sorted({node.op_type for node in graph.graph.node})
            unexpected = set(operators) - ALLOWED_OPERATORS
            if unexpected or any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
                raise RuntimeError(f"Unexpected export operators: {sorted(unexpected)}")
            checks = []
            if check_ort:
                pairs = {}
                for index, fixture_seed in enumerate((seed, (seed + 1) & 0xFFFFFFFF)):
                    check_pixels = fixture_input(size, fixture_seed)
                    with torch.inference_mode():
                        expected = model(torch.from_numpy(check_pixels)).numpy()
                    pairs.update({f"input{index}": check_pixels, f"expected{index}": expected,
                                  f"seed{index}": fixture_seed})
                pairs_path = stage / "native-check.npz"
                np.savez(pairs_path, **pairs)
                checks = native_parity(model_path, pairs_path, threads)
                pairs_path.unlink()
            manifest["cases"].append({
                "size": size, "inputShape": list(input_shape), "outputShape": list(output_shape),
                "model": describe_file(model_path),
                "input": write_f32(stage / f"input-{size}.f32", pixels),
                "expected": write_f32(stage / f"expected-{size}.f32", logits),
                "operators": operators, "pythonParity": checks,
            })
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        # Commit the manifest last so a partial copy cannot advertise a new
        # complete bundle. Runtime SHA-256 checks reject mixed bundle files.
        for path in stage.iterdir():
            if path.name != "manifest.json":
                shutil.copyfile(path, out / path.name)
        shutil.copyfile(stage / "manifest.json", out / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, help="export a checkpoint from the current training command")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-ort-check", action="store_true",
                        help="explicitly skip Python ONNX Runtime parity; record not_run in manifest")
    args = parser.parse_args()
    try:
        manifest = export_smoke(args.out, args.seed, threads=args.threads, check_ort=not args.skip_ort_check,
                                checkpoint=args.checkpoint)
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"{manifest['status'].capitalize()} model: {manifest['parameterCount']:,} parameters; CPU export, opset {OPSET}.")
    for case in manifest["cases"]:
        if case["pythonParity"]:
            error = max(item["maxAbsError"] for item in case["pythonParity"])
            note = f"two-input CPU parity passed (max abs {error:.3g})"
        else:
            note = "Python ONNX Runtime parity NOT RUN; browser parity still required"
        print(f"G={case['size']}: {case['model']['bytes']:,} ONNX bytes; {note}.")
    print(f"Bundle: {args.out.resolve() / 'manifest.json'}")
    if manifest["status"] == "untrained":
        print("This model is untrained. Benchmark execution/parity only; do not use it for live payload decisions.")
    else:
        print("Trained checkpoint exported. Runtime parity and held-out real-device recovery still require validation.")


if __name__ == "__main__":
    main()
