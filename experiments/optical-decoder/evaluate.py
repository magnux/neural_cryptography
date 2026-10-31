#!/usr/bin/env python3
"""Evaluate held-out physical labels and canonical JS CRC/RS payload recovery.

Predicted logits never receive expected bytes. Those bytes are compared only
after the canonical decoder accepts/rejects a candidate. Both methods observe
the same single capture; this report does not measure temporal throughput.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import Dataset, check_checkpoint_overlap

ROOT = Path(__file__).resolve().parents[2]


def masked_loss(logits, labels, mask, colors: int):
    if colors not in (4, 16) or not torch.any(mask):
        raise ValueError("A supported palette and nonempty data-cell mask are required")
    loss = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    return loss[mask].mean()


def canonical_recovery(dataset, selected, predictions, work_dir: Path, node="node", *, session_groups=None):
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
        if (not isinstance(values, list) or len(values) != len(wanted)
                or {row.get("id") for row in values} != {sample.id for sample in wanted}):
            raise RuntimeError("Canonical recovery returned a different sample set")
        # Attach reporting identities only after independent canonical decoding.
        # They never enter the prediction file or the decoder's evidence.
        by_id = {sample.id: sample for sample in wanted}
        for row in values:
            metadata = by_id[row["id"]].metadata
            row["sessionId"] = metadata["sessionId"]
            row["phase"] = metadata["info"].get("phase", 0)
            row["group"] = (session_groups[metadata["sessionId"]]
                if session_groups is not None and metadata["source"] == "real"
                else f"source:{metadata['source']}")
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


def select_samples(samples, maximum=0):
    """Deterministic round-robin coverage of sessions, sources, and profiles.

    IDs within each group are already hash-derived but sorted explicitly so file
    order cannot change the checkpoint-selection set. Unlimited means all rows.
    """
    if not isinstance(maximum, int) or maximum < 0:
        raise ValueError("max_samples must be nonnegative")
    if not maximum or maximum >= len(samples):
        return list(samples)
    groups = defaultdict(list)
    for sample in samples:
        row = sample.metadata
        groups[(row["sessionId"], row["size"], row["colors"], row["source"])].append(sample)
    keys = sorted(groups, key=lambda key: hashlib.sha256(repr(key).encode()).digest())
    for values in groups.values():
        values.sort(key=lambda sample: sample.id)
    selected, offset = [], 0
    while len(selected) < maximum:
        for key in keys:
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                if len(selected) == maximum:
                    break
        offset += 1
    return selected


def summarize_groups(per_session, session_groups=None, *, exact=True):
    """Aggregate session measurements, retaining macro group selection separately.

    Each group's CE/error is cell weighted and its recovery is frame weighted.
    Checkpoint selection gives each real group equal weight, regardless of
    its number of frames; synthetic rows never dilute that selection objective.
    """
    grouped, sourced = defaultdict(list), defaultdict(list)
    for session_id, row in per_session.items():
        source = row["source"]
        name = (session_groups[session_id] if session_groups is not None and source == "real"
                else f"source:{source}")
        grouped[name].append(row)
        sourced[source].append(row)

    def combine(rows):
        samples = sum(row["samples"] for row in rows)
        cells = sum(row["dataCells"] for row in rows)
        wrong = sum(row["wrongCells"] for row in rows)
        result = {"samples": samples, "dataCells": cells, "wrongCells": wrong,
                  "loss": sum(row["loss"]*row["dataCells"] for row in rows)/cells,
                  "cellErrorRate": wrong/cells}
        for name in ("learned", "classical"):
            frames = sum(row[name+"ExactFrames"] for row in rows) if exact else None
            result[name+"ExactFrames"] = frames
            result[name+"ExactRate"] = frames/samples if exact else None
        return result

    groups = {name: combine(rows) for name, rows in sorted(grouped.items())}
    sources = {name: combine(rows) for name, rows in sorted(sourced.items())}
    real_groups = sorted({session_groups[session_id] for session_id, row in per_session.items()
                          if row["source"] == "real"}) if session_groups is not None else []
    if real_groups:
        count = len(real_groups)
        score = {"method": "group-balanced-real", "groups": real_groups,
                 "exactRate": sum(groups[name]["learnedExactRate"] for name in real_groups)/count if exact else None,
                 "loss": sum(groups[name]["loss"] for name in real_groups)/count,
                 "cellErrorRate": sum(groups[name]["cellErrorRate"] for name in real_groups)/count}
    else:
        all_rows = combine(list(per_session.values()))
        score = {"method": "global", "groups": [], "exactRate": all_rows["learnedExactRate"],
                 "loss": all_rows["loss"], "cellErrorRate": all_rows["cellErrorRate"]}
    return {"groups": groups, "sources": sources, "selectionScore": score}


def evaluate_model(model, dataset: Dataset, *, device="cpu", max_samples=0, node="node", exact=True,
                   work_dir=None, session_groups=None):
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    selected = select_samples(dataset.samples, max_samples)
    if session_groups is not None:
        from sampling import validate_session_groups
        available_groups = validate_session_groups(dataset.samples, session_groups)
        selected_groups = validate_session_groups(selected, session_groups)
        omitted_groups = available_groups-selected_groups
        if omitted_groups:
            raise ValueError(f"Capped evaluation omits real groups {sorted(omitted_groups)}; increase max_samples or use 0")
    directory = Path(work_dir or Path.cwd()/"local-results/optical-decoder").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous_mode = model.training
    model.eval()
    model.to(device)
    confusion = np.zeros((16, 16), dtype=np.int64)
    bins = np.zeros((15, 3), dtype=np.float64)
    loss_sum, cells = 0., 0
    timings, predictions, profile_counts = [], {}, {}
    per_session = {}
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
                    sample_loss = float(masked_loss(output, target, valid, colors))
                    loss_sum += sample_loss*count
                    cells += count
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    chosen = probabilities.argmax(axis=0)[mask]
                    truth = labels[mask]
                    confidence = probabilities.max(axis=0)[mask]
                    session_id = sample.metadata["sessionId"]
                    group = per_session.setdefault(session_id, {"samples": 0, "dataCells": 0, "wrongCells": 0,
                        "source": sample.metadata["source"], "lossSum": 0.,
                        "phases": {}, "learnedExactFrames": 0 if exact else None,
                        "classicalExactFrames": 0 if exact else None})
                    if group["source"] != sample.metadata["source"]:
                        raise ValueError("A receiver session cannot mix real and synthetic sources")
                    group["samples"] += 1
                    group["dataCells"] += count
                    group["wrongCells"] += int(np.count_nonzero(chosen != truth))
                    group["lossSum"] += sample_loss*count
                    phase = str(sample.metadata["info"].get("phase", 0))
                    group["phases"][phase] = group["phases"].get(phase, 0)+1
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
            recovery = canonical_recovery(dataset, selected, predictions, temp, node,
                                          session_groups=session_groups) if exact else None
    finally:
        model.train(previous_mode)
    selected_by_id = {sample.id: sample for sample in selected}
    if recovery:
        for row in recovery["samples"]:
            group = per_session[selected_by_id[row["id"]].metadata["sessionId"]]
            group["learnedExactFrames"] += int(row["learned"]["exact"])
            group["classicalExactFrames"] += int(row["classical"]["exact"])
    for group in per_session.values():
        group["cellErrorRate"] = group["wrongCells"]/group["dataCells"]
        group["loss"] = group.pop("lossSum")/group["dataCells"]
        for name in ("learned", "classical"):
            group[name+"ExactRate"] = group[name+"ExactFrames"]/group["samples"] if exact else None
    summaries = summarize_groups(per_session, session_groups, exact=exact)
    available_sessions = {sample.metadata["sessionId"] for sample in dataset.samples}
    nonempty = bins[:, 0] > 0
    ece = np.sum(np.abs(bins[nonempty, 1]-bins[nonempty, 2])) / cells
    ordered = sorted(timings)
    return {"format": "optical-decoder-evaluation", "version": 1,
            "samples": len(selected), "availableSamples": len(dataset.samples),
            "manifestSha256": dataset.manifest_hash, "profiles": profile_counts,
            "selection": {"method": "session-profile-round-robin", "maximum": max_samples,
                "availableSessions": len(available_sessions), "selectedSessions": len(per_session),
                "omittedSessions": sorted(available_sessions-set(per_session)),
                "note": "A capped evaluation is a subset; use --max-samples 0 for all held-out samples."},
            "sessions": per_session, **summaries,
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
    parser.add_argument("--session-groups", type=Path, help="explicit receiver-session groups for balanced real-direction metrics")
    parser.add_argument("--node", default="node")
    parser.add_argument("--out", type=Path, default=Path("local-results/optical-decoder/evaluation.json"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1–64")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    try:
        from checkpoint import load_model_checkpoint
        dataset = Dataset(args.dataset, args.split)
        session_groups = None
        if args.session_groups:
            from sampling import load_session_groups
            session_groups = load_session_groups(args.session_groups)
        model, checkpoint = load_model_checkpoint(args.checkpoint)
        check_checkpoint_overlap(checkpoint, dataset, args.split)
        result = evaluate_model(model, dataset, device=device, max_samples=args.max_samples,
                                node=args.node, work_dir=args.out.resolve().parent, session_groups=session_groups)
        result["modelProfile"] = model.profile
        result["split"] = args.split
        result["checkpoint"] = str(args.checkpoint.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+"\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(json.dumps({"samples": result["samples"], "cellErrorRate": result["cellErrorRate"],
                      "loss": result["loss"], "recovery": result["recovery"]["summary"],
                      "groups": result["groups"], "selectionScore": result["selectionScore"],
                      "report": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
