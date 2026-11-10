#!/usr/bin/env python3
"""Evaluate held-out optical matrices and, where defined, packet recovery.

Neural-QR datasets measure exact logical data cells and a separate recorded-error
RS/CRC replay benchmark. The OBM experiment uses actual canonical JS packet
recovery. Replay applies measured errors to encoded test packets; it does not
claim that raw collection symbols contain parity or checksums. Predicted logits
never receive expected bytes. This report does not measure throughput.
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
from bursts import BurstIndex, EVALUATION_POLICY, EVALUATION_SEED
from contract import FRAME_COUNTS, MAX_FRAMES
from reporting import compact_report, evaluation_document, lean_json
from packet_replay import validate_rs_parity
from model import W_TUBE_PROFILES

ROOT = Path(__file__).resolve().parents[2]


def require_implemented_model_contract(dataset, model=None):
    """Check raster/target semantics before sampling or invoking the network.

    The optional first call checks the dataset alone; training checks again
    after constructing/loading its requested model. A shared raster size is
    insufficient: OBM packet targets and Neural-QR logical matrices differ.
    """
    format_ = getattr(dataset, "dataset_format", "optical-decoder-dataset")
    ppc = getattr(dataset, "pixels_per_cell", 4)
    version = getattr(dataset, "dataset_version", 1)
    contracts = {("optical-decoder-dataset", 1, 4),
                 ("neural-qr-dataset", 1, 2), ("neural-qr-dataset", 2, 4)}
    if (format_, version, ppc) not in contracts:
        raise ValueError("Unsupported dataset format/version/pixels-per-cell contract")
    if model is None:
        return
    profile = model if isinstance(model, str) else getattr(model, "profile", None)
    expected = ("neural-qr-dataset", 2, 4) if profile in W_TUBE_PROFILES else (
        ("neural-qr-dataset", 1, 2) if profile == "multires" else
        ("optical-decoder-dataset", 1, 4))
    if (format_, version, ppc) != expected:
        raise ValueError("W-tube requires Neural-QR v2 at 4 pixels per cell, as does VW; multires requires "
                         "Neural-QR v1 at 2 pixels per cell; U-tube profiles require OBM datasets")
    if hasattr(model, "pixels_per_cell") and model.pixels_per_cell != ppc:
        raise ValueError("Model raster scale disagrees with its dataset contract")


def masked_loss(logits, labels, mask, colors: int):
    if colors not in (4, 16) or not torch.any(mask):
        raise ValueError("A supported palette and nonempty data-cell mask are required")
    loss = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    return loss[mask].mean()


def canonical_recovery(dataset, selected, predictions, work_dir: Path, node="node", *, session_groups=None):
    rows = []
    label_spaces = {"logical" if getattr(sample, "phase_grouped", False) else "physical"
                    for sample in selected}
    if len(label_spaces) != 1:
        raise ValueError("Recovery candidates cannot mix logical and physical output labels")
    label_space = label_spaces.pop()
    members = {sample.id: tuple(getattr(sample, "members", (sample,))) for sample in selected}
    counts = {len(group) for group in members.values()}
    if len(counts) != 1 or any(count not in FRAME_COUNTS for count in counts):
        raise ValueError(f"Recovery candidates must all contain the same 1 to {MAX_FRAMES} captures")
    frames = counts.pop()
    for index, manifest in enumerate(sorted({sample.manifest for sample in selected})):
        wanted = [sample for sample in selected if sample.manifest == manifest]
        prediction_file = work_dir / f"predictions-{index}.json"
        prediction_file.write_text(json.dumps({"framesPerInput": frames, "outputLabelSpace": label_space,
            "samples": [{"id": sample.id,
            "memberIds": [member.id for member in members[sample.id]],
            "logits": predictions[sample.id].relative_to(work_dir).as_posix()} for sample in wanted]}))
        process = subprocess.run([node, str(ROOT/"scripts/corrector-dataset.mjs"), "evaluate",
                                  "--dataset", str(manifest.parent), "--predictions", str(prediction_file)],
                                 capture_output=True, text=True, timeout=max(60, len(wanted)*3))
        if process.returncode:
            raise RuntimeError(f"Canonical recovery bridge failed: {process.stderr[-2000:]}")
        result = json.loads(process.stdout)
        values = result.get("samples")
        if (not isinstance(values, list) or len(values) != len(wanted)
                or {row.get("id") for row in values} != {sample.id for sample in wanted}):
            raise RuntimeError("Canonical recovery returned a different sample set")
        if any(row.get("memberIds") != [member.id for member in members[row["id"]]] for row in values):
            raise RuntimeError("Canonical recovery returned different capture groups")
        # Attach reporting identities only after independent canonical decoding.
        # They never enter the prediction file or the decoder's evidence.
        by_id = {sample.id: sample for sample in wanted}
        for row in values:
            metadata = by_id[row["id"]].metadata
            row["sessionId"] = metadata["sessionId"]
            row["phase"] = None if label_space == "logical" else metadata["info"].get("phase", 0)
            row["phases"] = sorted({member.metadata["info"].get("phase", 0)
                                    for member in members[row["id"]]})
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
    return {"summary": summary, "samples": rows, "outputLabelSpace": label_space,
            "framesPerInput": frames, "inputCaptures": len(rows)*frames,
            "observationBudget": (f"The same set of {frames} distinct captures per method and candidate; "
                                  "classical scores are averaged over those captures. "
                                  "Candidates are nonoverlapping; no throughput claim.")}


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


def _evaluate_candidates(model, dataset, candidates, burst_stats, *, frames, device="cpu",
                         max_samples=0, node="node", exact=True, work_dir=None, session_groups=None):
    if not candidates:
        if "frameCapacity" in burst_stats:
            raise ValueError(f"No complete {burst_stats['frameCapacity']}-capture evaluation blocks; "
                             "nested four-phase evaluation needs two captures of every phase")
        raise ValueError(f"No complete {frames}-frame evaluation groups; collect more same-phase captures")
    selected = select_samples(candidates, max_samples)
    if session_groups is not None:
        from sampling import validate_session_groups
        available_groups = validate_session_groups(dataset.samples, session_groups)
        selected_groups = validate_session_groups(selected, session_groups)
        omitted_groups = available_groups-selected_groups
        if omitted_groups:
            raise ValueError(f"Evaluation omits real groups {sorted(omitted_groups)}; collect complete bursts and increase max_samples or use 0")
    directory = Path(work_dir or Path.cwd()/"local-results/neural-corrector").resolve()
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
                    phase = ("0+1+2+3" if getattr(sample, "phase_grouped", False)
                             else str(sample.metadata["info"].get("phase", 0)))
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
    return {"format": "neural-corrector-evaluation", "version": 1,
            "samples": len(selected), "availableSamples": len(candidates),
            "outputLabelSpace": "logical" if getattr(selected[0], "phase_grouped", False) else "physical",
            "framesPerInput": frames, "inputCaptures": len(selected)*frames,
            "availableInputCaptures": len(dataset.samples), "burstGrouping": burst_stats,
            "manifestSha256": dataset.manifest_hash, "profiles": profile_counts,
            "selection": {"method": f"{burst_stats.get('method', EVALUATION_POLICY)}/nonoverlapping/session-profile-round-robin",
                "seed": EVALUATION_SEED, "maximum": max_samples,
                "availableSessions": len(available_sessions), "selectedSessions": len(per_session),
                "omittedSessions": sorted(available_sessions-set(per_session)),
                "note": "One fixed seeded member permutation per burst forms complete, nonoverlapping groups. "
                        "Remainders are reported, never duplicated; use --max-samples 0 for all complete groups."},
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


def _combine_frame_counts(reports, *, frame_capacity, session_groups, exact):
    """Summarize repeated nested tests without calling them independent captures.

    Each K keeps its own complete report. Selection gives every observation
    count equal weight, and each real direction equal weight within that count.
    """
    counts = sorted(reports, key=int)
    values = [reports[key] for key in counts]
    first = values[0]
    per_session = {}
    for report in values:
        for session, row in report["sessions"].items():
            total = per_session.setdefault(session, {"samples": 0, "dataCells": 0,
                "wrongCells": 0, "source": row["source"], "lossSum": 0., "phases": {},
                "learnedExactFrames": 0 if exact else None,
                "classicalExactFrames": 0 if exact else None})
            if total["source"] != row["source"]:
                raise ValueError("A receiver session cannot mix real and synthetic sources")
            for key in ("samples", "dataCells", "wrongCells"):
                total[key] += row[key]
            total["lossSum"] += row["loss"]*row["dataCells"]
            for phase, amount in row["phases"].items():
                total["phases"][phase] = total["phases"].get(phase, 0)+amount
            if exact:
                for name in ("learned", "classical"):
                    total[name+"ExactFrames"] += row[name+"ExactFrames"]
    for row in per_session.values():
        row["loss"] = row.pop("lossSum")/row["dataCells"]
        row["cellErrorRate"] = row["wrongCells"]/row["dataCells"]
        for name in ("learned", "classical"):
            row[name+"ExactRate"] = row[name+"ExactFrames"]/row["samples"] if exact else None
    summaries = summarize_groups(per_session, session_groups, exact=exact)
    scores = [report["selectionScore"] for report in values]
    balanced = all(score["method"] == "group-balanced-real" for score in scores)
    summaries["selectionScore"] = {
        "method": "frame-count-group-balanced-real" if balanced else "frame-count-balanced",
        "groups": scores[0]["groups"], "frameCounts": [int(key) for key in counts],
        "exactRate": sum(score["exactRate"] for score in scores)/len(scores) if exact else None,
        "loss": sum(score["loss"] for score in scores)/len(scores),
        "cellErrorRate": sum(score["cellErrorRate"] for score in scores)/len(scores)}
    recovery = None
    if exact:
        recovery = {"summary": {}, "samples": [], "framesPerInput": None,
                    "outputLabelSpace": first["outputLabelSpace"],
                    "frameCapacity": frame_capacity, "actualFrameCounts": [int(key) for key in counts],
                    "inputCaptures": sum(report["inputCaptures"] for report in values),
                    "uniqueInputCaptures": first["samples"]*frame_capacity,
                    "observationBudget": "Each frame count uses the same distinct captures for learned and classical decoding. "
                        "Counts are nested within the same capacity-sized blocks, so observations repeat across presets. "
                        "Summed cases are not independent captures or a throughput measurement."}
        for name in ("classical", "learned"):
            fields = ("frames", "rawCRCFrames", "hardCRCFrames", "acceptedFrames",
                      "exactFrames", "wrongAcceptedFrames")
            combined = {field: sum(report["recovery"]["summary"][name][field] for report in values)
                        for field in fields}
            combined["exactRate"] = combined["exactFrames"]/combined["frames"]
            recovery["summary"][name] = combined
        for key, report in reports.items():
            recovery["samples"].extend({**row, "actualFrames": int(key)} for row in report["recovery"]["samples"])
    goal_rows = {}
    for key, report in reports.items():
        group = report["groups"].get("phone-to-pc")
        goal_rows[key] = ({"available": bool(exact), "samples": group["samples"], "exactFrames": group["learnedExactFrames"],
                           "exactRate": group["learnedExactRate"],
                           "achieved": group["learnedExactRate"] >= .5 if exact else None}
                          if group else {"available": False, "samples": 0, "exactFrames": None,
                                         "exactRate": None, "achieved": None})
    goal_available = all(row["available"] for row in goal_rows.values())
    goal = {"direction": "phone-to-pc", "targetExactRate": .5,
            "available": goal_available, "byFrameCount": goal_rows,
            "exactRate": sum(row["exactRate"] for row in goal_rows.values())/len(goal_rows) if goal_available else None,
            "achievedForAllFrameCounts": all(row["achieved"] for row in goal_rows.values()) if goal_available else None,
            "note": "Measured exact CRC/RS recovery on held-out phone-to-PC cases; requires explicit direction groups and exact evaluation."}
    confusion = np.sum([report["confusion"] for report in values], axis=0)
    profiles = {key: sum(report["profiles"].get(key, 0) for report in values)
                for key in sorted(set().union(*(report["profiles"] for report in values)))}
    bins = []
    for index in range(len(first["calibrationBins"])):
        rows = [report["calibrationBins"][index] for report in values]
        count = sum(row["count"] for row in rows)
        bins.append({"count": count,
                     "confidence": sum(row["count"]*(row["confidence"] or 0.) for row in rows)/count if count else None,
                     "accuracy": sum(row["count"]*(row["accuracy"] or 0.) for row in rows)/count if count else None})
    cells = sum(report["dataCells"] for report in values)
    return {"format": "neural-corrector-evaluation", "version": 1,
            "outputLabelSpace": first["outputLabelSpace"],
            "samples": sum(report["samples"] for report in values),
            "observationCases": sum(report["samples"] for report in values),
            "uniqueCandidates": first["samples"],
            "availableSamples": sum(report["availableSamples"] for report in values),
            "uniqueAvailableCandidates": first["availableSamples"],
            "framesPerInput": frame_capacity, "frameCapacity": frame_capacity,
            "actualFrameCounts": [int(key) for key in counts],
            "inputCaptures": sum(report["inputCaptures"] for report in values),
            "uniqueInputCaptures": first["samples"]*frame_capacity,
            "availableInputCaptures": first["availableInputCaptures"],
            "burstGrouping": {"method": "fixed-seeded-phase-pairs/nested-frame-counts",
                "seed": EVALUATION_SEED, "frameCapacity": frame_capacity,
                "actualFrameCounts": [int(key) for key in counts],
                "uniqueCandidates": first["samples"],
                "note": "The same two-capture-per-phase blocks and fixed paired slots are reused for all nested frame counts. "
                        "See byFrameCount for unused-slot and remainder accounting."},
            "manifestSha256": first["manifestSha256"], "profiles": profiles,
            "selection": {**first["selection"],
                "method": "fixed-seeded-phase-pairs/nested-frame-counts/session-profile-round-robin",
                "note": "One fixed shuffled four-phase capacity block defines every 4/5/6/7/8 preset. "
                        "max_samples caps the same candidate IDs per preset. Cases repeat across presets."},
            "sessions": per_session, **summaries, "phoneToPCGoal": goal,
            "dataCells": cells, "loss": sum(report["loss"] for report in values)/len(values),
            "cellErrorRate": sum(report["cellErrorRate"] for report in values)/len(values),
            "confusion": confusion.tolist(), "calibrationBins": bins,
            "expectedCalibrationError": sum(row["count"]*abs(row["confidence"]-row["accuracy"])
                                                for row in bins if row["count"])/cells,
            "inferenceMs": {"byFrameCount": {key: report["inferenceMs"] for key, report in reports.items()},
                            "note": "PyTorch inference plus CPU copy per preset; not browser latency."},
            "recovery": recovery, "byFrameCount": reports}


def evaluate_model(model, dataset: Dataset, *, device="cpu", max_samples=0, node="node", exact=True,
                   work_dir=None, session_groups=None, detailed=False, rs_parity=32):
    require_implemented_model_contract(dataset, model)
    validate_rs_parity(rs_parity)
    if type(detailed) is not bool:
        raise ValueError("detailed must be a boolean")
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    if getattr(model, "profile", None) in ("multires", *W_TUBE_PROFILES):
        from multires_evaluation import evaluate_matrices
        result = evaluate_matrices(model, dataset, device=device, max_samples=max_samples,
                                   session_groups=session_groups, detailed=detailed,
                                   rs_parity=rs_parity, node=node, work_dir=work_dir)
        return {**result, "reportDetail": "detailed"} if detailed else compact_report(result)
    if getattr(dataset, "dataset_format", "optical-decoder-dataset") != "optical-decoder-dataset":
        raise ValueError("Neural-QR datasets require the matching multires, W-tube or VW model")
    supported = getattr(model, "supported_sizes", (64, 128))
    if any(sample.shape[0] not in supported for sample in dataset.samples):
        raise ValueError(f"Model supports only matrix sizes {supported}")
    frames = getattr(model, "frames", 1)
    variable = getattr(model, "variable_frames", False)
    options = dict(device=device, max_samples=max_samples, node=node, exact=exact,
                   work_dir=work_dir, session_groups=session_groups)
    if not variable:
        candidates, stats = BurstIndex(dataset.samples, frames).eval_samples()
        result = _evaluate_candidates(model, dataset, candidates, stats, frames=frames, **options)
        return {**result, "reportDetail": "detailed"} if detailed else compact_report(result)
    minimum = model.min_frames
    index = BurstIndex(dataset.samples, frames, min_frames=minimum, presence=True)
    reports = {}
    candidate_ids = None
    for count in range(minimum, frames+1):
        candidates, stats = index.eval_samples(actual_frames=count)
        identities = tuple(sample.id for sample in candidates)
        if candidate_ids is not None and identities != candidate_ids:
            raise ValueError("Variable-count evaluation must retain the same candidate IDs at every count")
        candidate_ids = identities
        reports[str(count)] = _evaluate_candidates(model, dataset, candidates, stats, frames=count, **options)
        reports[str(count)]["frameCapacity"] = frames
    result = _combine_frame_counts(reports, frame_capacity=frames, session_groups=session_groups, exact=exact)
    return {**result, "reportDetail": "detailed"} if detailed else compact_report(result)


def format_evaluation_summary(report):
    """Human-readable aggregate lines, without expanding the leaf table on stdout."""
    def metrics(row):
        text = f"CE {row['loss']:.4f} | CER {100*row['cellErrorRate']:.2f}%"
        if "matrixAccuracy" in row:
            exact, count = row["matrixAccuracy"]["exactMatrices"], row["matrixAccuracy"]["matrices"]
        elif "learnedExactFrames" in row:
            exact, count = row["learnedExactFrames"], row["samples"]
        else:
            learned = (row.get("recovery") or {}).get("summary", {}).get("learned")
            exact, count = (learned["exactFrames"], learned["frames"]) if learned else (None, None)
        if exact is not None and count:
            text += f" | exact {exact}/{count} ({100*exact/count:.2f}%)"
        replay = row.get("rsReplay")
        if replay is not None:
            text += f" | RS+CRC replay {replay['exactCases']}/{replay['eligibleCases']}"
        return text

    lines = [f"Evaluation | {metrics(report)}"]
    if "rsReplay" in report:
        replay = report["rsReplay"]
        lines.append(f"  Recorded-error replay: RS decoded {replay['rsDecodedCases']}/{replay['eligibleCases']} | "
                     f"CRC accepted {replay['crcAcceptedCases']}/{replay['eligibleCases']} | "
                     f"wrong accepted {replay['wrongAcceptedCases']} | "
                     f"ineligible {replay['ineligibleCases']} (not measured packet reception)")
    if "evaluationPopulation" in report:
        grids = ", ".join(f"G{size}" for size in report["primarySizes"])
        lines.append(f"  Population: {report['evaluationPopulation']} | primary grids: {grids}")
    if "selectionScore" in report:
        score = report["selectionScore"]
        lines.append(f"  Selection: {100*score['exactRate']:.2f}% [{score['method']}]")
    for grid, row in sorted(report.get("bySize", {}).items(), key=lambda item: int(item[0])):
        lines.append(f"  G{grid} | {metrics(row)}")
    for label, key in (("Direction", "groups"), ("Origin", "origins")):
        for name, row in sorted(report.get(key, {}).items()):
            lines.append(f"  {label} {name} | {metrics(row)}")
    by_k = report.get("byFrameCount", {})
    if by_k:
        counts = []
        for k, row in sorted(by_k.items(), key=lambda item: int(item[0])):
            exact = row.get("matrixAccuracy", {}).get("exactRate")
            if exact is None:
                exact = (row.get("recovery") or {}).get("summary", {}).get("learned", {}).get("exactRate")
            counts.append(f"K{k}: {100*exact:.1f}%" if exact is not None else f"K{k}: n/a")
        lines.append("  Exact by frames | "+" | ".join(counts))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="maximum complete input groups; 0 evaluates all")
    parser.add_argument("--session-groups", type=Path, help="explicit receiver-session groups for balanced real-direction metrics")
    parser.add_argument("--node", default="node")
    parser.add_argument("--rs-parity", type=int, default=32,
                        help="Neural-QR recorded-error replay: even RS parity bytes/block (2..64), 0 disables; default 32")
    parser.add_argument("--out", type=Path, default=Path("local-results/neural-corrector/evaluation.json"))
    parser.add_argument("--detailed-report", type=Path,
                        help="optional separate report with session/capture rows, confusion and calibration arrays")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1–64")
    if args.detailed_report and args.detailed_report.resolve() == args.out.resolve():
        parser.error("--detailed-report must differ from the compact --out report")
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
        if model.profile in ("multires", *W_TUBE_PROFILES):
            unseen = {sample.shape[0] for sample in dataset.samples} - set(checkpoint["training"]["trainedSizes"])
            if unseen:
                raise ValueError(f"Checkpoint anchors have not been trained for grids {sorted(unseen)}")
        result = evaluate_model(model, dataset, device=device, max_samples=args.max_samples,
                                node=args.node, work_dir=args.out.resolve().parent, session_groups=session_groups,
                                detailed=args.detailed_report is not None, rs_parity=args.rs_parity)
        result["modelProfile"] = model.profile
        result["split"] = args.split
        result["checkpoint"] = str(args.checkpoint.resolve())
        if args.detailed_report:
            args.detailed_report.parent.mkdir(parents=True, exist_ok=True)
            args.detailed_report.write_text(json.dumps(result, indent=2)+"\n")
        result = compact_report(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(lean_json(evaluation_document(result)))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print("\n".join([*format_evaluation_summary(result), f"Report: {args.out.resolve()}"]))


if __name__ == "__main__":
    main()
