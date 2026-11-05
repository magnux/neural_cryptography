"""Held-out Neural-QR logical matrix accuracy, independent of packet recovery.

No CRC/RS decoder is applicable to these raw-symbol collection targets. Target
labels are read only after inference inputs have been constructed and are used
only for scoring; they never alter a prediction or select its capture members.
"""

from collections import defaultdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from bursts import BurstIndex, EVALUATION_SEED, VARIABLE_EVALUATION_POLICY
from sampling import infer_session_groups, validate_session_groups


def counter():
    return {"samples": 0, "lossSum": 0., "wrongCells": 0, "dataCells": 0,
            "exactMatrices": 0, "inputCaptures": 0}


def add(counter, *, loss, wrong, cells, captures):
    counter["samples"] += 1
    counter["lossSum"] += loss
    counter["wrongCells"] += wrong
    counter["dataCells"] += cells
    counter["exactMatrices"] += int(wrong == 0)
    counter["inputCaptures"] += captures


def combine(rows):
    result = counter()
    for row in rows:
        for name in result:
            result[name] += row[name]
    return result


def summarize(row):
    n, cells = row["samples"], row["dataCells"]
    return {"samples": n, "inputCaptures": row["inputCaptures"],
            "loss": row["lossSum"]/n, "wrongCells": row["wrongCells"], "dataCells": cells,
            "cellErrorRate": row["wrongCells"]/cells,
            "matrixAccuracy": {"matrices": n, "exactMatrices": row["exactMatrices"],
                               "exactRate": row["exactMatrices"]/n}}


def _selection(leaves):
    """Equal G weight, equal real acquisition-group weight, then equal K weight.

    Each leaf's CE is already a mean over examples, not pooled pixels, so G256
    cannot outweigh G8 merely by having more cells. Synthetic validation may be
    reported but cannot dilute the held-out real selection objective.
    """
    real = {key: row for key, row in leaves.items() if key[1] == "real"}
    if not real:
        raise ValueError("Multires checkpoint selection requires held-out real captures")
    sizes = sorted({key[0] for key in real})
    scores = []
    for size in sizes:
        names = sorted({key[2] for key in real if key[0] == size})
        by_group = []
        for name in names:
            rows = [summarize(row) for key, row in sorted(real.items()) if key[0] == size and key[2] == name]
            by_group.append([np.mean([row["matrixAccuracy"]["exactRate"] for row in rows]),
                             np.mean([row["loss"] for row in rows]),
                             np.mean([row["cellErrorRate"] for row in rows])])
        scores.append(np.mean(by_group, axis=0))
    exact, loss, cer = np.mean(scores, axis=0).tolist()
    return {"method": "size-group-frame-balanced-real", "sizes": sizes,
            "groups": sorted({key[2] for key in real}), "frameCounts": [4, 5, 6, 7, 8],
            "exactRate": exact, "loss": loss, "cellErrorRate": cer}


def _summary_with_size_macro(leaves):
    result = summarize(combine(leaves.values()))
    grids = defaultdict(list)
    for key, row in leaves.items():
        grids[key[0]].append(row)
    result["sizeMacroCellErrorRate"] = float(np.mean([
        summarize(combine(rows))["cellErrorRate"] for rows in grids.values()]))
    return result


def _breakdown(leaves):
    summary = _summary_with_size_macro(leaves)
    for label, position in (("sources", 1), ("groups", 2)):
        names = sorted({key[position] for key in leaves})
        summary[label] = {str(name): _summary_with_size_macro({
            key: row for key, row in leaves.items() if key[position] == name}) for name in names}
    return summary


def evaluate_matrices(model, dataset, *, device="cpu", max_samples=0, session_groups=None,
                      detailed=False):
    from evaluate import select_samples  # Shared unbiased diagnostic subset selection.
    if dataset.dataset_format != "neural-qr-dataset" or model.profile != "multires":
        raise ValueError("Neural-QR matrix evaluation requires the multires model and Neural-QR dataset")
    groups = infer_session_groups(dataset.samples) if session_groups is None else session_groups
    validate_session_groups(dataset.samples, groups)
    index = BurstIndex(dataset.samples, 8, min_frames=4, presence=True)
    leaves = defaultdict(counter)
    counts, by_count, timings, rows = {}, {}, [], []
    first_ids, used_ids, available_ids = None, set(), set()
    previous_mode = model.training
    model.to(device).eval()
    try:
        with torch.inference_mode():
            for count in range(4, 9):
                candidates, stats = index.eval_samples(actual_frames=count)
                if not candidates:
                    raise ValueError("Multires validation needs at least two distinct captures of each of four phases")
                selected = select_samples(candidates, max_samples)
                identities = tuple(sample.id for sample in selected)
                if first_ids is not None and identities != first_ids:
                    raise ValueError("Every K preset must use the same seeded capacity blocks")
                first_ids = identities
                available = {(s.shape[0], s.metadata["source"], groups[s.metadata["sessionId"]])
                             for s in dataset.samples if s.metadata["source"] == "real"}
                chosen = {(s.shape[0], s.metadata["source"], groups[s.metadata["sessionId"]])
                          for s in selected if s.metadata["source"] == "real"}
                if available-chosen:
                    raise ValueError("Evaluation cap omits real resolution/group combinations; use --max-samples 0")
                for sample in selected:
                    pixels, labels, mask = sample.tensors()
                    values = torch.from_numpy(pixels[None]).to(device)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    output = model(values)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    timings.append((time.perf_counter()-start)*1000)
                    if tuple(output.shape) != (1, 16, *labels.shape) or not torch.isfinite(output).all():
                        raise ValueError("Corrector produced invalid matrix logits")
                    target = torch.from_numpy(labels[None]).to(device)
                    valid = torch.from_numpy(mask[None]).to(device)
                    loss = float(F.cross_entropy(output, target, reduction="none")[valid].mean().cpu())
                    wrong = int(((output.argmax(1) != target) & valid).sum().cpu())
                    cells = int(mask.sum())
                    source, size = sample.metadata["source"], sample.shape[0]
                    group = groups[sample.metadata["sessionId"]] if source == "real" else "source:synthetic"
                    add(leaves[(size, source, group, count)], loss=loss, wrong=wrong, cells=cells, captures=count)
                    used_ids.update(member.id for member in sample.members)
                    if detailed:
                        rows.append({"id": sample.id, "memberIds": [m.id for m in sample.members],
                                     "sessionId": sample.metadata["sessionId"], "source": source,
                                     "size": size, "group": group, "frames": count,
                                     "loss": loss, "wrongCells": wrong, "dataCells": cells,
                                     "exact": wrong == 0})
                available_ids.update(member.id for sample in candidates for member in sample.members)
                current = {key: value for key, value in leaves.items() if key[3] == count}
                by_count[str(count)] = {**_breakdown(current), "framesPerInput": count,
                                       "burstGrouping": stats}
                counts[str(count)] = len(selected)
    finally:
        model.train(previous_mode)
    sizes = sorted({key[0] for key in leaves})
    by_size = {}
    for size in sizes:
        subset = {key: row for key, row in leaves.items() if key[0] == size}
        by_size[str(size)] = {**_breakdown(subset), "byFrameCount": {
            str(k): _breakdown({key: row for key, row in subset.items() if key[3] == k})
            for k in range(4, 9)}}
    result = {"format": "neural-corrector-evaluation", "version": 1,
              "modelProfile": "multires", "datasetFormat": "neural-qr-dataset",
              "outputLabelSpace": "logical", "metric": "exact-masked-logical-matrix",
              "note": "cellErrorRate pools data cells; sizeMacroCellErrorRate averages per-grid CER over represented grids. Exact means every data cell is correct, excluding fixed references. No CRC, error correction or packet recovery is measured. K presets reuse nested captures and are not independent trials.",
              "manifestSha256": dataset.manifest_hash, "framesPerInput": 8,
              "frameCapacity": 8, "actualFrameCounts": [4, 5, 6, 7, 8],
              "pixelsPerCell": 2, "profiles": {str(size): sum(v["samples"] for k, v in leaves.items() if k[0] == size) for size in sizes},
              **_breakdown(leaves), "selectionScore": _selection(leaves),
              "bySize": by_size, "byFrameCount": by_count,
              "observationCases": sum(counts.values()), "uniqueCandidates": counts["4"],
              "uniqueInputCaptures": len(used_ids), "availableInputCaptures": len(available_ids),
              "burstGrouping": {"method": VARIABLE_EVALUATION_POLICY, "seed": EVALUATION_SEED,
                                "frameCapacity": 8, "actualFrameCounts": [4, 5, 6, 7, 8]},
              "inferenceMs": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)),
                              "note": "PyTorch forward latency; no browser or transport throughput claim."}}
    if detailed:
        result["cases"] = rows
    return result
