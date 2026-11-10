"""Held-out raw matrix accuracy and separate recorded-error RS/CRC replay.

Collection symbols have no packet envelope. Target labels score inference and
define diagnostic error patterns; they never repair a prediction, select its
captures or enter the packet decoder. Replay is not measured packet reception.
"""

from collections import defaultdict
from contextlib import ExitStack
import time

import numpy as np
import torch
from torch.nn import functional as F

from bursts import BurstIndex, EVALUATION_SEED, VARIABLE_EVALUATION_POLICY
from sampling import infer_session_groups, validate_session_groups
from reporting import evaluation_table
from packet_replay import (REPLAY_COUNTERS, ReplayBatch, add_replay,
                           replay_policy, replay_summary, validate_rs_parity)
from model import W_TUBE_PROFILES


def counter():
    return {"samples": 0, "lossSum": 0., "wrongCells": 0, "dataCells": 0,
            "exactMatrices": 0, "inputCaptures": 0,
            **dict.fromkeys(REPLAY_COUNTERS, 0)}


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
    result = {"samples": n, "inputCaptures": row["inputCaptures"],
            "loss": row["lossSum"]/n, "wrongCells": row["wrongCells"], "dataCells": cells,
            "cellErrorRate": row["wrongCells"]/cells,
            "matrixAccuracy": {"matrices": n, "exactMatrices": row["exactMatrices"],
                               "exactRate": row["exactMatrices"]/n}}
    replay = replay_summary(row)
    if replay is not None:
        result["rsReplay"] = replay
    return result


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


def _primary_wtube_leaves(leaves):
    """Select native real evidence when available, keeping auxiliary results separate."""
    real = {key: row for key, row in leaves.items() if key[1] == "real"}
    if not real:
        raise ValueError("W-tube checkpoint selection requires held-out real captures")
    native = {key: row for key, row in real.items() if key[4] == "native"}
    return (native, "native") if native else (real, "auxiliary-only")


def _by_size(leaves):
    result = {}
    for size in sorted({key[0] for key in leaves}):
        subset = {key: row for key, row in leaves.items() if key[0] == size}
        result[str(size)] = {**_breakdown(subset), "byFrameCount": {
            str(k): _breakdown({key: row for key, row in subset.items() if key[3] == k})
            for k in range(4, 9)}}
    return result


def evaluate_matrices(model, dataset, *, device="cpu", max_samples=0, session_groups=None,
                      detailed=False, rs_parity=32, node="node", work_dir=None):
    from evaluate import select_samples, require_implemented_model_contract
    validate_rs_parity(rs_parity)
    require_implemented_model_contract(dataset, model)
    if dataset.dataset_format != "neural-qr-dataset" or model.profile not in ("multires", *W_TUBE_PROFILES):
        raise ValueError("Neural-QR matrix evaluation requires its matching multires, W-tube or VW model")
    wtube = model.profile in W_TUBE_PROFILES
    if wtube and any(sample.metadata.get("origin") not in ("native", "resampled", "header-g64")
                     for sample in dataset.samples):
        raise ValueError("W-tube evaluation requires explicit native/resampled/header-g64 origin metadata")
    if any(sample.shape[0] not in model.supported_sizes for sample in dataset.samples):
        raise ValueError(f"Model supports only matrix sizes {model.supported_sizes}")
    groups = infer_session_groups(dataset.samples) if session_groups is None else session_groups
    validate_session_groups(dataset.samples, groups)
    index = BurstIndex(dataset.samples, 8, min_frames=4, presence=True)
    leaves = defaultdict(counter)
    counts, by_count, timings, rows = {}, {}, [], []
    first_ids, used_ids, available_ids = None, set(), set()
    used_by_origin, available_by_origin = defaultdict(set), defaultdict(set)
    replay_keys = []
    with ExitStack() as stack:
        replay = (stack.enter_context(ReplayBatch(rs_parity, node=node, work_dir=work_dir))
                  if rs_parity else None)
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
                    available = {(s.shape[0], s.metadata["source"], groups[s.metadata["sessionId"]],
                                  s.metadata.get("origin") if wtube else None)
                                 for s in dataset.samples if s.metadata["source"] == "real"}
                    chosen = {(s.shape[0], s.metadata["source"], groups[s.metadata["sessionId"]],
                               s.metadata.get("origin") if wtube else None)
                              for s in selected if s.metadata["source"] == "real"}
                    if available-chosen:
                        raise ValueError("Evaluation cap omits real resolution/group/origin combinations; use --max-samples 0")
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
                        predicted = output.argmax(1)
                        wrong = int(((predicted != target) & valid).sum().cpu())
                        cells = int(mask.sum())
                        source, size = sample.metadata["source"], sample.shape[0]
                        group = groups[sample.metadata["sessionId"]] if source == "real" else "source:synthetic"
                        origin = sample.metadata.get("origin")
                        key = (size, source, group, count, origin) if wtube else (size, source, group, count)
                        add(leaves[key], loss=loss, wrong=wrong, cells=cells, captures=count)
                        if replay is not None:
                            replay.append(labels, predicted[0].cpu().numpy(), mask)
                            replay_keys.append(key)
                        used_ids.update(member.id for member in sample.members)
                        if source == "real":
                            used_by_origin[origin].update(member.id for member in sample.members)
                        if detailed:
                            rows.append({"id": sample.id, "memberIds": [m.id for m in sample.members],
                                         "sessionId": sample.metadata["sessionId"], "source": source,
                                         "size": size, "group": group, "frames": count,
                                         **({"origin": origin} if wtube else {}),
                                         "loss": loss, "wrongCells": wrong, "dataCells": cells,
                                         "exact": wrong == 0})
                    available_ids.update(member.id for sample in candidates for member in sample.members)
                    for sample in candidates:
                        if sample.metadata["source"] == "real":
                            available_by_origin[sample.metadata.get("origin")].update(m.id for m in sample.members)
                    current = {key: value for key, value in leaves.items() if key[3] == count}
                    by_count[str(count)] = {**_breakdown(current), "framesPerInput": count,
                                           "burstGrouping": stats}
                    counts[str(count)] = len(selected)
        finally:
            model.train(previous_mode)
        if replay is not None:
            for position, result in enumerate(replay.results()):
                add_replay(leaves[replay_keys[position]], result)
                if detailed:
                    rows[position]["rsReplay"] = {k: v for k, v in result.items() if k != "id"}
            for k in range(4, 9):
                by_count[str(k)].update(_breakdown({key: row for key, row in leaves.items() if key[3] == k}))
    all_leaves, extra = leaves, {}
    if wtube:
        leaves, population = _primary_wtube_leaves(all_leaves)
        selected_origins = sorted({key[4] for key in leaves})
        origins = {}
        for origin in sorted({key[4] for key in all_leaves if key[1] == "real"}):
            subset = {key: row for key, row in all_leaves.items() if key[4] == origin and key[1] == "real"}
            origins[origin] = {**_summary_with_size_macro(subset),
                               "sizes": sorted({key[0] for key in subset})}
        extra = {"evaluationPopulation": population, "nativeAvailable": population == "native",
                 "origins": origins, "evaluatedCases": sum(counts.values()),
                 "evaluatedSizes": sorted({key[0] for key in all_leaves}),
                 "primarySizes": sorted({key[0] for key in leaves})}
        synthetic = {key: row for key, row in all_leaves.items() if key[1] == "synthetic"}
        if synthetic:
            extra["syntheticValidation"] = _summary_with_size_macro(synthetic)
        by_count = {str(k): {**_breakdown({key: row for key, row in leaves.items() if key[3] == k}),
                            "framesPerInput": k} for k in range(4, 9)}
        counts = {str(k): by_count[str(k)]["samples"] for k in range(4, 9)}
        used_ids = set().union(*(used_by_origin[origin] for origin in selected_origins))
        available_ids = set().union(*(available_by_origin[origin] for origin in selected_origins))
    sizes = sorted({key[0] for key in leaves})
    score = _selection(leaves)
    if wtube:
        score.update(originScope=extra["evaluationPopulation"],
                     nativeAvailable=extra["nativeAvailable"], origins=selected_origins)
    result = {"format": "neural-corrector-evaluation", "version": 1,
              "modelProfile": model.profile, "datasetFormat": "neural-qr-dataset",
              "datasetVersion": dataset.dataset_version,
              "outputLabelSpace": "logical", "metric": "exact-masked-logical-matrix",
              "note": "cellErrorRate pools data cells; sizeMacroCellErrorRate averages per-grid CER over represented grids. Exact means every data cell is correct, excluding fixed references, without error correction. RS/CRC replay is a separate recorded-error diagnostic, not measured packet reception. K presets reuse nested captures and are not independent trials.",
              "manifestSha256": dataset.manifest_hash, "framesPerInput": 8,
              "frameCapacity": 8, "actualFrameCounts": [4, 5, 6, 7, 8],
              "pixelsPerCell": model.pixels_per_cell, "profiles": {str(size): sum(v["samples"] for k, v in leaves.items() if k[0] == size) for size in sizes},
              "metricTable": evaluation_table(all_leaves), "rsReplayPolicy": replay_policy(rs_parity),
              **_breakdown(leaves), "selectionScore": score,
              "bySize": _by_size(leaves), "byFrameCount": by_count, **extra,
              "observationCases": sum(counts.values()), "uniqueCandidates": counts["4"],
              "uniqueInputCaptures": len(used_ids), "availableInputCaptures": len(available_ids),
              "burstGrouping": {"method": VARIABLE_EVALUATION_POLICY, "seed": EVALUATION_SEED,
                                "frameCapacity": 8, "actualFrameCounts": [4, 5, 6, 7, 8]},
              "inferenceMs": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)),
                              "note": "PyTorch forward latency; no browser or transport throughput claim."}}
    if wtube:
        result["note"] += (" Headline and grid/K tables use native real captures when available; otherwise they use explicitly labelled auxiliary real evidence. "
                           "Origins are reported separately, and primarySizes lists the grids covered by the headline.")
    if detailed:
        result["cases"] = rows
    return result
