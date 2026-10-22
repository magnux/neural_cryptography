#!/usr/bin/env python3
"""Evaluate held-out physical labels and canonical JS CRC/RS payload recovery.

Predicted logits never receive expected bytes. Those bytes are compared only
after the canonical decoder accepts/rejects a candidate. Both methods observe
the same single capture; this report does not measure temporal throughput.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import Dataset, check_checkpoint_overlap
from model import OpticalDecoder

ROOT = Path(__file__).resolve().parents[2]


def masked_loss(logits, labels, mask, colors: int):
    if colors not in (4, 16) or not torch.any(mask):
        raise ValueError("A supported palette and nonempty data-cell mask are required")
    loss = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    return loss[mask].mean()


def canonical_recovery(dataset, selected, predictions, work_dir: Path, node="node"):
    rows = []
    for index, manifest in enumerate(sorted({sample.manifest for sample in selected})):
        wanted = [sample for sample in selected if sample.manifest == manifest]
        prediction_file = work_dir / f"predictions-{index}.json"
        prediction_file.write_text(json.dumps({"samples": [{"id": sample.id,
            "logits": predictions[sample.id].relative_to(work_dir).as_posix()} for sample in wanted]}))
        process = subprocess.run([node, str(ROOT/"scripts/decoder-dataset.mjs"), "evaluate",
                                  "--dataset", str(manifest.parent), "--predictions", str(prediction_file)],
                                 capture_output=True, text=True, timeout=max(60, len(wanted)*3))
        if process.returncode:
            raise RuntimeError(f"Canonical recovery bridge failed: {process.stderr[-2000:]}")
        result = json.loads(process.stdout)
        values = result.get("samples")
        if not isinstance(values, list) or {row.get("id") for row in values} != {sample.id for sample in wanted}:
            raise RuntimeError("Canonical recovery returned a different sample set")
        rows.extend(values)
    summary = {}
    for name in ("classical", "learned"):
        values = [row[name] for row in rows]
        summary[name] = {"frames": len(values),
            "rawCRCFrames": sum(bool(row.get("rawCRC")) for row in values),
            "hardCRCFrames": sum(bool(row.get("hardCRC")) for row in values),
            "acceptedFrames": sum(bool(row.get("accepted")) for row in values),
            "exactFrames": sum(bool(row.get("exact")) for row in values),
            "wrongAcceptedFrames": sum(bool(row.get("wrongAccepted")) for row in values)}
        summary[name]["exactRate"] = summary[name]["exactFrames"] / len(values)
    return {"summary": summary, "samples": rows,
            "observationBudget": "One identical input capture per method; no temporal fusion or throughput claim."}


def evaluate_model(model, dataset: Dataset, *, device="cpu", max_samples=0, node="node", exact=True,
                   work_dir=None):
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    selected = dataset.samples[:max_samples] if max_samples else dataset.samples
    if len(selected) > 4096:
        raise ValueError("Evaluation is bounded to 4096 captures; set --max-samples explicitly")
    directory = Path(work_dir or Path.cwd()/"local-results/optical-decoder").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous_mode = model.training
    model.eval()
    model.to(device)
    confusion = np.zeros((16, 16), dtype=np.int64)
    bins = np.zeros((15, 3), dtype=np.float64)
    loss_sum, cells = 0., 0
    timings, predictions, profile_counts = [], {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix="eval-", dir=directory) as temporary:
            temp = Path(temporary)
            with torch.inference_mode():
                for index, sample in enumerate(selected):
                    pixels, labels, mask = sample.tensors()
                    _, colors = sample.shape
                    input_tensor = torch.from_numpy(pixels[None]).to(device)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    output = model(input_tensor)
                    values = output.detach().cpu().numpy()[0]
                    elapsed = (time.perf_counter()-started)*1000
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite predictions for {sample.id}")
                    timings.append(elapsed)
                    logits = output[:, :colors]
                    target = torch.from_numpy(labels[None]).to(device)
                    valid = torch.from_numpy(mask[None]).to(device)
                    count = int(mask.sum())
                    loss_sum += float(masked_loss(output, target, valid, colors))*count
                    cells += count
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    chosen = probabilities.argmax(axis=0)[mask]
                    truth = labels[mask]
                    confidence = probabilities.max(axis=0)[mask]
                    np.add.at(confusion, (truth, chosen), 1)
                    calibration_bin = np.minimum(14, (confidence*15).astype(int))
                    np.add.at(bins[:, 0], calibration_bin, 1)
                    np.add.at(bins[:, 1], calibration_bin, confidence)
                    np.add.at(bins[:, 2], calibration_bin, chosen == truth)
                    key = f"{sample.metadata['size']}/{colors}/{sample.metadata['source']}"
                    profile_counts[key] = profile_counts.get(key, 0)+1
                    if exact:
                        path = temp/f"prediction-{index}.f32"
                        path.write_bytes(np.ascontiguousarray(values, dtype="<f4").tobytes())
                        predictions[sample.id] = path
            recovery = canonical_recovery(dataset, selected, predictions, temp, node) if exact else None
    finally:
        model.train(previous_mode)
    nonempty = bins[:, 0] > 0
    ece = np.sum(np.abs(bins[nonempty, 1]-bins[nonempty, 2])) / cells
    ordered = sorted(timings)
    return {"format": "optical-decoder-evaluation", "version": 1,
            "samples": len(selected), "availableSamples": len(dataset.samples),
            "manifestSha256": dataset.manifest_hash, "profiles": profile_counts,
            "dataCells": cells, "loss": loss_sum/cells,
            "cellErrorRate": 1-float(np.trace(confusion))/cells,
            "confusion": confusion.tolist(), "expectedCalibrationError": float(ece),
            "calibrationBins": [{"count": int(n), "confidence": float(c/n) if n else None,
                                "accuracy": float(a/n) if n else None} for n, c, a in bins],
            "inferenceMs": {"p50": float(np.median(ordered)),
                            "p95": ordered[min(len(ordered)-1, int(np.ceil(.95*len(ordered)))-1)],
                            "note": "PyTorch inference plus CPU output copy; not browser latency."},
            "recovery": recovery}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--node", default="node")
    parser.add_argument("--out", type=Path, default=Path("local-results/optical-decoder/evaluation.json"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1–64")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    try:
        from checkpoint import load_checkpoint
        dataset = Dataset(args.dataset, args.split)
        model = OpticalDecoder()
        checkpoint = load_checkpoint(args.checkpoint, model)
        check_checkpoint_overlap(checkpoint, dataset, args.split)
        result = evaluate_model(model, dataset, device=device, max_samples=args.max_samples,
                                node=args.node, work_dir=args.out.resolve().parent)
        result["split"] = args.split
        result["checkpoint"] = str(args.checkpoint.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(json.dumps({"samples": result["samples"], "cellErrorRate": result["cellErrorRate"],
                      "loss": result["loss"], "recovery": result["recovery"]["summary"],
                      "report": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
