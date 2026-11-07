#!/usr/bin/env python3
"""Train an optical classifier with ordinary masked cross-entropy.

Plain PyTorch, float32, AdamW. Optional target pairing places real captures and
synthetic renders of the same matrix in each optimizer update. Validation uses
held-out sessions: exact logical matrices for Neural-QR multires, canonical
packet recovery for the OBM experiment; never the final test split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from torch.nn import functional as F
from augmentation import RealAugmentation, PhaseAugmentation
from bursts import BurstIndex, EVALUATION_POLICY, EVALUATION_SEED, VARIABLE_EVALUATION_POLICY
from contract import FRAME_COUNTS, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX
from checkpoint import load_model_checkpoint, initialize_multires
from data import DATASET_FORMAT, Dataset, assert_disjoint, check_checkpoint_overlap, digest
from evaluate import evaluate_model, require_implemented_model_contract
from reporting import compact_report
from model import DEFAULT_MODEL_PROFILE, MODEL_PROFILES, MULTIRES_SIZES, NeuralCorrector, initialize_frames
from sampling import BalancedSampler, load_session_groups, validate_session_groups, infer_session_groups


def training_defaults(initialize=None, steps=None, learning_rate=None):
    default_lr = 5e-5 if initialize is not None else 1e-4
    return steps if steps is not None else 32_000, learning_rate if learning_rate is not None else default_lr


def training_loss(logits, labels, mask, colors):
    """Mean data-cell CE per input example, then equal weight per batch member."""
    counts = mask.flatten(1).sum(1)
    if colors not in (4, 16) or not torch.all(counts > 0):
        raise ValueError("A supported palette and nonempty data-cell mask per input example are required")
    cells = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    per_frame = (cells*mask).flatten(1).sum(1)/counts
    return per_frame.mean(), per_frame


def training_cell_counts(logits, labels, mask, colors):
    """Per-example wrong/data counts from this forward pass, excluding pilots."""
    with torch.no_grad():
        wrong = ((logits[:, :colors].argmax(dim=1) != labels) & mask).flatten(1).sum(1)
        return wrong, mask.flatten(1).sum(1)


def metric_counter():
    return {"lossSum": 0., "samples": 0, "inputCaptures": 0, "wrongCells": 0, "dataCells": 0}


def summarize_training_counters(counters, *, sources=False):
    """Stable JSON/console order, independent of shuffled batch membership."""
    names = ([name for name in ("real", "synthetic") if name in counters] if sources else sorted(counters))
    return {name: {"samples": row["samples"], "inputCaptures": row["inputCaptures"],
                   "meanCE": row["lossSum"]/row["samples"],
                   "cellErrorRate": row["wrongCells"]/row["dataCells"],
                   "wrongCells": row["wrongCells"], "dataCells": row["dataCells"]}
            for name in names for row in (counters[name],)}


def summarize_training_grids(counters, source_counters):
    """Same-resolution real/synthetic summaries, with numeric grid ordering."""
    mixed = summarize_training_counters(counters)
    return {size: {**mixed[size],
                   "sources": summarize_training_counters(source_counters.get(size, {}), sources=True)}
            for size in sorted(mixed, key=int)}


def size_macro_cer(by_size):
    """Equal weight for each represented grid, independent of its cell count."""
    return (sum(row["cellErrorRate"] for row in by_size.values()) / len(by_size)
            if by_size else None)


def training_source_summaries(interval):
    result = summarize_training_counters(interval["sources"], sources=True)
    if "sizeSources" in interval:
        for name, row in result.items():
            grids = {size: summarize_training_counters(values, sources=True)[name]
                     for size, values in interval["sizeSources"].items() if name in values}
            row["sizeMacroCellErrorRate"] = size_macro_cer(grids)
    return result


def format_multires_grids(training, validation):
    """One concise row per grid; never compare mixed train to real validation."""
    def metrics(row, *, train=False, exact=False):
        if row is None:
            return "n/a"
        text = f"CE {row['meanCE' if train else 'loss']:.4f} CER {row['cellErrorRate']:.2%}"
        if exact:
            text += f" exact {row['matrixAccuracy']['exactRate']:.1%}"
        return text
    lines = []
    for size in sorted(set(training) | set(validation), key=int):
        train_sources = training.get(size, {}).get("sources", {})
        val_sources = validation.get(size, {}).get("sources", {})
        lines.append(f"  G{size:<3} | train real {metrics(train_sources.get('real'), train=True)}"
                     f" | train synthetic {metrics(train_sources.get('synthetic'), train=True)}"
                     f" | val real {metrics(val_sources.get('real'), exact=True)}")
    return lines


def format_training_metrics(interval):
    """Only current training-interval metrics; never cached validation values."""
    if not interval["updates"]:
        return "no optimizer updates"
    text = (f"CE {interval['loss']/interval['updates']:.4f}"
            f" CER {interval['wrongCells']/interval['dataCells']:.2%} (cell)")
    if "sizeSources" in interval:
        macro = size_macro_cer(summarize_training_counters(interval["sizes"]))
        if macro is not None:
            text += f" CER(size) {macro:.2%}"
    for values in (training_source_summaries(interval),
                   summarize_training_counters(interval["groups"])):
        if values:
            text += " | " + "; ".join(f"{name} CE {row['meanCE']:.4f} CER {row['cellErrorRate']:.2%} (cell)"
                                       for name, row in values.items())
    return text


def exact_metrics(report):
    """Uniform display counters without claiming raw Neural-QR targets have CRCs."""
    if "matrixAccuracy" in report:
        row = report["matrixAccuracy"]
        return {"exactFrames": row["exactMatrices"], "frames": row["matrices"], "exactRate": row["exactRate"]}
    return report["recovery"]["summary"]["learned"]


def format_validation_metrics(report):
    learned = exact_metrics(report)
    selection = report["selectionScore"]
    text = (f"CE {report['loss']:.4f} CER {report['cellErrorRate']:.2%} (cell)"
            + (f" CER(size) {report['sizeMacroCellErrorRate']:.2%}" if "sizeMacroCellErrorRate" in report else "")
            + f" | exact {learned['exactFrames']}/{learned['frames']} ({learned['exactRate']:.2%})"
            f" | selection {selection['exactRate']:.2%} [{selection['method']}]")
    if "byFrameCount" in report:
        text += " | " + " ".join(f"K{k}:{exact_metrics(row)['exactRate']:.1%}"
                                    for k, row in sorted(report["byFrameCount"].items(), key=lambda item: int(item[0])))
    return text


def format_validation_groups(report):
    return "; ".join(f"{name} CE {row['loss']:.4f} CER {row['cellErrorRate']:.2%} (cell)"
                     f" exact {row['matrixAccuracy']['exactRate'] if 'matrixAccuracy' in row else row['learnedExactRate']:.1%}"
                     for name, row in sorted(report["groups"].items()))


def save_checkpoint(path: Path, model, training: dict):
    value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
             "modelState": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
             "training": training}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def draw_training_bursts(batches, sampler, burst_index, rng):
    """Keep each shuffled real/synthetic pair's observation budget identical."""
    if not burst_index.presence:
        return [[burst_index.train_sample(anchor, rng) for anchor in anchors] for anchors in batches]
    if (len(sampler.last_pair_ids) != len(batches)
            or any(len(anchors) != len(ids) for anchors, ids in zip(batches, sampler.last_pair_ids))):
        raise ValueError("Sampler pair identities do not match this optimizer update")
    paired_counts, result = {}, []
    for anchors, pair_ids in zip(batches, sampler.last_pair_ids):
        samples = []
        for anchor, pair_id in zip(anchors, pair_ids):
            if pair_id is None:
                count = rng.randint(burst_index.min_frames, burst_index.frames)
            else:
                if pair_id not in paired_counts:
                    paired_counts[pair_id] = rng.randint(burst_index.min_frames, burst_index.frames)
                count = paired_counts[pair_id]
            samples.append(burst_index.train_sample(anchor, rng, actual_frames=count))
        result.append(samples)
    return result


def training_tensors(samples, augmentation, phase_augmentation=None):
    """Augment real observations before zero filling and adding validity planes."""
    triples = [sample.raw_tensors() for sample in samples]
    if samples[0].slots is None:
        pixels = augmentation(np.stack([row[0] for row in triples]),
                              [sample.metadata["source"] for sample in samples])
    else:
        pixels = np.stack([sample.pack_pixels(augmentation(row[0][None],
                           [sample.metadata["source"]])[0])
                           for sample, row in zip(samples, triples)])
    labels, masks = np.stack([row[1] for row in triples]), np.stack([row[2] for row in triples])
    if phase_augmentation is not None:
        if not all(sample.phase_grouped for sample in samples):
            raise ValueError("Phase augmentation requires phase-grouped inputs")
        return phase_augmentation(pixels, labels, masks)
    return pixels, labels, masks


def run_training(train: Dataset, validation: Dataset, out: Path, *, steps=None, batch_size=4,
                 accumulate=4, learning_rate=None, device="cpu", seed=7, eval_every=1000,
                 validation_samples=0, real_fraction=.5, initialize=None, node="node", model_profile=None,
                 session_groups=None, weight_decay=.01, paired_targets=False, frames=None, no_progress=False,
                 real_noise_std=.008, real_tint_stops=.06, real_exposure_stops=.08,
                 phase_flip_probability=.25, no_phase_permutations=False, show_goal=False):
    # New recording imports are usable for synthetic preparation, but must not
    # silently reach the earlier 2-pixel multires network or its optimizer.
    require_implemented_model_contract(train)
    require_implemented_model_contract(validation)
    steps, learning_rate = training_defaults(initialize, steps, learning_rate)
    if frames is not None and (type(frames) is not int or frames not in FRAME_COUNTS):
        raise ValueError("Input frame count must be an integer from 1 through 64")
    integers = ((steps, 1, 1_000_000), (batch_size, 1, 128), (accumulate, 1, 128),
                (seed, 0, 0xffffffff), (eval_every, 1, 1_000_000),
                (validation_samples, 0, 1_000_000_000))
    if (any(type(value) is not int or not low <= value <= high for value, low, high in integers)
            or any(isinstance(value, bool) for value in (learning_rate, weight_decay))
            or not np.isfinite(learning_rate) or not 0 < learning_rate <= .1
            or not np.isfinite(weight_decay) or not 0 <= weight_decay <= 1
            or not isinstance(paired_targets, bool) or type(show_goal) is not bool):
        raise ValueError("Invalid training budget, learning rate, weight decay, or pairing setting")
    real_augmentation = RealAugmentation(seed=seed, noise_std=real_noise_std,
                                        tint_stops=real_tint_stops, exposure_stops=real_exposure_stops)
    if any(s.metadata["split"] != "train" for s in train.samples):
        raise ValueError("Only train samples may be fitted")
    if any(s.metadata["split"] != "validation" for s in validation.samples):
        raise ValueError("Only validation samples may select checkpoints")
    assert_disjoint(train, validation)
    multires_data = train.dataset_format == "neural-qr-dataset"
    if validation.dataset_format != train.dataset_format:
        raise ValueError("Training and validation must use the same experiment dataset format")
    if multires_data and session_groups is None:
        session_groups = infer_session_groups(train.samples+validation.samples)
    if session_groups is not None:
        validate_session_groups(train.samples, session_groups)
        validate_session_groups(validation.samples, session_groups)
    # Optional acquisition directions are explicit; never infer them from IDs.
    sampler_groups = session_groups if session_groups is not None else {
        s.metadata["sessionId"]: "all-real" for s in train.samples if s.metadata["source"] == "real"}
    sampler = BalancedSampler(train.samples, sampler_groups, seed=seed, real_fraction=real_fraction,
                              paired_targets=paired_targets, require_all_groups=not multires_data)
    if paired_targets and batch_size*accumulate % 2:
        raise ValueError("Paired targets require an even effective batch size")
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty run directory; existing runs are never overwritten")
    if model_profile is not None and model_profile not in MODEL_PROFILES:
        raise ValueError("Unknown neural-corrector model profile")
    torch.manual_seed(seed)
    previous, transfer = None, None
    if initialize is None:
        model = NeuralCorrector(model_profile or DEFAULT_MODEL_PROFILE, frames=frames)
    else:
        model, previous = load_model_checkpoint(initialize)
        if model_profile == "multires" and model.profile == "U-tube-2x-attn":
            model, transfer = initialize_multires(model)
        elif model_profile is not None and model.profile != model_profile:
            raise ValueError("Requested --model differs from the initialized checkpoint profile; train that architecture from scratch")
        if frames is not None:
            model = initialize_frames(model, frames)
    if (model.profile == "multires") != multires_data:
        raise ValueError("multires requires Neural-QR 2-pixel-cell data; U-tube profiles require their 4-pixel-cell experiment data")
    variable_frames = getattr(model, "variable_frames", False)
    minimum_frames = getattr(model, "min_frames", model.frames)
    phase_augmentation = PhaseAugmentation(seed=seed, flip_probability=phase_flip_probability,
                                           permutations=not no_phase_permutations) if variable_frames else None
    frame_counts = list(range(minimum_frames, model.frames+1))
    if any(sample.shape[0] not in model.supported_sizes for sample in train.samples+validation.samples):
        raise ValueError(f"Model {model.profile} supports only target grid sizes {model.supported_sizes}")
    burst_index = BurstIndex(train.samples, model.frames, min_frames=minimum_frames, presence=variable_frames)
    burst_index.validate_training()
    burst_rng = random.Random(seed ^ 0x42555253)
    train_sessions, train_payloads = set(train.session_ids), set(train.payload_ids)
    validation_sessions, validation_payloads = set(validation.session_ids), set(validation.payload_ids)
    if previous is not None:
        check_checkpoint_overlap(previous, validation, "validation")
        ancestry = previous["training"]
        if (train.session_ids & set(ancestry["validationSessionIds"])
                or train.payload_ids & set(ancestry["validationPayloadIds"])):
            raise ValueError("Fine-tuning data overlaps previous checkpoint-selection data")
        train_sessions.update(ancestry["trainSessionIds"])
        train_payloads.update(ancestry["trainPayloadIds"])
        validation_sessions.update(ancestry["validationSessionIds"])
        validation_payloads.update(ancestry["validationPayloadIds"])
    training_sizes = sorted({sample.shape[0] for sample in train.samples})
    if multires_data and {sample.shape[0] for sample in validation.samples} - set(training_sizes):
        raise ValueError("Validation grids must have training data for their independent anchors")
    # A new anchor is not labeled trained until an optimizer update sees its G.
    learned_sizes = set(previous["training"].get("trainedSizes", [])) if previous and transfer is None else set()
    out.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    options = {"requestedSteps": steps, "batchSize": batch_size,
               "gradientAccumulation": accumulate, "initialLearningRate": learning_rate,
               "weightDecay": weight_decay, "realFraction": real_fraction,
               "pairedTargets": paired_targets, "framesPerInput": model.frames,
               "inputLayout": model.input_layout, "minFramesPerInput": minimum_frames,
               "frameCounts": frame_counts, "frameCountPolicy": "uniform-4-8" if variable_frames else "fixed",
               "modelProfile": model.profile, "networkInputClip": [NETWORK_INPUT_MIN, NETWORK_INPUT_MAX],
               "realAugmentation": real_augmentation.settings(),
               "phaseAugmentation": phase_augmentation.settings() if phase_augmentation else None,
               "loss": "mean-masked-cross-entropy",
               "evalEvery": eval_every, "validationSamples": validation_samples,
               "showGoal": show_goal,
               "evaluationGrouping": VARIABLE_EVALUATION_POLICY if variable_frames else EVALUATION_POLICY,
               "evaluationSeed": EVALUATION_SEED,
               "sampling": "paired-targets" if paired_targets else "source-group-session-payload-phase"}
    provenance = {"seed": seed, "samples": len(train.samples), "batchSize": batch_size,
                  "gradientAccumulation": accumulate, "device": str(device), "realFraction": real_fraction,
                  "pairedTargets": paired_targets, "framesPerInput": model.frames, "selection": "validation-exact", "datasetFormat": train.dataset_format,
                  "inputLayout": model.input_layout, "minFramesPerInput": minimum_frames,
                  "frameCounts": frame_counts, "frameCountPolicy": "uniform-4-8" if variable_frames else "fixed",
                  "modelProfile": model.profile, "networkInputClip": [NETWORK_INPUT_MIN, NETWORK_INPUT_MAX],
                  "trainManifestSha256": train.manifest_hash, "validationManifestSha256": validation.manifest_hash,
                  "trainSessionIds": sorted(train_sessions), "trainPayloadIds": sorted(train_payloads),
                  "validationSessionIds": sorted(validation_sessions), "validationPayloadIds": sorted(validation_payloads),
                  "settings": options}
    if multires_data:
        geometry = {"pixelsPerCell": 2, "supportedSizes": list(MULTIRES_SIZES)}
        options.update({**geometry, "trainingSizes": training_sizes,
                        "sampling": "uniform-size/source-group-session-payload-phase",
                        "selectionMetric": "size-group-frame-balanced-real/exact-logical-data-matrix"})
        provenance.update(geometry)
        architecture = {"anchor": model.anchor_metadata, "bottleneck": model.bottleneck_metadata}
        options.update(architecture)
        provenance.update(architecture)
        options["metricAggregation"] = {
            "cellErrorRate": "sum wrong data cells / sum data cells",
            "sizeMacroCellErrorRate": "mean of per-grid cellErrorRate over represented grids",
            "gridSources": "separate real and synthetic metrics within each grid"}
        if transfer is not None:
            options["warmStartTransfer"] = transfer
    if variable_frames:
        phase_metadata = {"outputLabelSpace": "logical", "phaseMasks": [0, 15, 7, 8]}
        options.update(phase_metadata)
        provenance.update(phase_metadata)
    if initialize is not None:
        provenance["initialCheckpointSha256"] = digest(Path(initialize))
    if session_groups is not None:
        group_hash = hashlib.sha256(json.dumps(session_groups, sort_keys=True).encode()).hexdigest()
        provenance["sessionGroupsSha256"] = group_hash
        # Exact grouping stays in the private local run/checkpoint, not exports.
        provenance["sessionGroups"] = session_groups
        (out/"session-groups.json").write_text(json.dumps({"format": "optical-decoder-session-groups",
            "version": 1, "sessions": session_groups}, indent=2)+"\n")
    (out/"settings.json").write_text(json.dumps(options, indent=2)+"\n")
    history, best = [], (-1., -float("inf"))
    selected_step, completed, stop_reason = None, 0, "budget"
    started = time.perf_counter()

    def evaluate(step, interval):
        nonlocal best, selected_step
        report = compact_report(evaluate_model(model, validation, device=device, max_samples=validation_samples,
                                node=node, work_dir=out, session_groups=session_groups))
        learned = exact_metrics(report)
        summary = report.get("recovery", {}).get("summary", {})
        if summary.get("learned", {}).get("wrongAcceptedFrames", 0):
            (out/"validation-failed.json").write_text(json.dumps(report, indent=2)+"\n")
            raise RuntimeError("Validation accepted incorrect payload bytes; inspect validation-failed.json")
        selection = report["selectionScore"]
        score = selection["exactRate"], -selection["loss"]
        # Step zero with fresh weights is diagnostic only, not a trained model.
        baseline_eligible = (previous is not None and transfer is None
                             and (not multires_data or set(training_sizes) <= learned_sizes))
        selected = score > best and (step > 0 or baseline_eligible)
        current_lr = optimizer.param_groups[0]["lr"]
        if selected:
            best, selected_step = score, step
            training = {**provenance, "steps": step, "selectionStep": step,
                        "baselineSelection": step == 0, "learningRate": current_lr,
                        "selectionAggregation": selection["method"],
                        "bestValidationExact": selection["exactRate"], "bestValidationLoss": selection["loss"]}
            if multires_data:
                training["trainedSizes"] = sorted(learned_sizes)
            save_checkpoint(out/"best.pt", model, training)
            (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
        source_metrics = training_source_summaries(interval)
        group_metrics = summarize_training_counters(interval["groups"])
        record = {"step": step, "trainLoss": interval["loss"]/interval["updates"] if interval["updates"] else None,
                  "trainCellErrorRate": interval["wrongCells"]/interval["dataCells"] if interval["dataCells"] else None,
                  "trainDataCells": interval["dataCells"], "trainWrongCells": interval["wrongCells"],
                  "trainSources": source_metrics, "trainGroups": group_metrics,
                  "sampling": interval["sampling"], "validationLoss": report["loss"],
                  "validationCellErrorRate": report["cellErrorRate"],
                  "validationExact": learned["exactRate"],
                  "validationSelection": selection, "validationGroups": report["groups"],
                  "selected": selected, "learningRate": current_lr,
                  "elapsedSeconds": time.perf_counter()-started}
        if "classical" in summary:
            record["classicalExact"] = summary["classical"]["exactRate"]
        if multires_data:
            record["trainBySize"] = summarize_training_grids(interval["sizes"], interval["sizeSources"])
            record["trainSizeMacroCellErrorRate"] = size_macro_cer(record["trainBySize"])
            record["validationSizeMacroCellErrorRate"] = report["sizeMacroCellErrorRate"]
            record["validationSources"] = report["sources"]
            record["validationBySize"] = report["bySize"]
        if variable_frames:
            record["trainFrameCounts"] = interval["frameCounts"]
            record["validationByFrameCount"] = report["byFrameCount"]
            if "phoneToPCGoal" in report:
                record["phoneToPCGoal"] = report["phoneToPCGoal"]
        history.append(record)
        lines = [f"Step {step}/{steps} | lr {current_lr:.2g}{' | best' if selected else ''}",
                 f"  Train | {format_training_metrics(interval)}",
                 f"  Val   | {format_validation_metrics(report)}"]
        if report["groups"]:
            lines.append(f"  Val groups | {format_validation_groups(report)}")
        if multires_data:
            lines.extend(format_multires_grids(record["trainBySize"], report["bySize"]))
        if variable_frames and show_goal and "phoneToPCGoal" in report:
            goal = report["phoneToPCGoal"]
            lines.append("  Goal phone→PC | " + " ".join(
                f"K{k}:{row['exactRate']:.1%}" if row['exactRate'] is not None else f"K{k}:n/a"
                for k, row in sorted(goal["byFrameCount"].items(), key=lambda item: int(item[0]))) + " / target 50%")
        tqdm.write("\n".join(lines), file=sys.stdout)
        target = "baseline-validation.json" if step == 0 else "final-validation.json"
        (out/target).write_text(json.dumps(report, indent=2)+"\n")

    def empty_interval():
        return {"loss": 0., "updates": 0, "wrongCells": 0, "dataCells": 0, "sources": {}, "groups": {}, "sizes": {},
                **({"sizeSources": {}} if multires_data else {}),
                "frameCounts": {str(count): 0 for count in frame_counts},
                "sampling": {"repeatedBursts": 0, "repeatedSamples": 0, "pairedTargets": 0}}

    def write_history():
        (out/"history.json").write_text(json.dumps({"format": "neural-corrector-training-run", "version": 1,
            "requestedSteps": steps, "completedSteps": completed, "selectedStep": selected_step,
            "stopReason": stop_reason, "device": str(device), "settings": options, "history": history,
            "note": "Training CE/CER include configured photometric and phase/spatial augmentation; validation inputs are unchanged. "
                    "CER(cell) pools data cells; CER(size) averages per-grid CER over represented grids. "
                    "Grid metrics separate real and synthetic training sources. "
                    "Validation selects checkpoints; untouched real test sessions establish generalization."}, indent=2)+"\n")

    def update_progress(progress, interval, refresh=False):
        progress.set_postfix_str(
            f"{format_training_metrics(interval)} | lr {optimizer.param_groups[0]['lr']:.2g}", refresh=refresh)

    try:
        with tqdm(total=steps, desc="Training", unit="update", dynamic_ncols=True,
                  mininterval=.5, file=sys.stdout, disable=no_progress or not sys.stdout.isatty()) as progress:
            evaluate(0, empty_interval())
            write_history()
            interval = empty_interval()
            for step in range(1, steps+1):
                optimizer.zero_grad(set_to_none=True)
                update_loss = 0.
                batches = draw_training_bursts(sampler.draw_update(batch_size, accumulate),
                                               sampler, burst_index, burst_rng)
                for samples in batches:
                    pixels, labels, mask = training_tensors(samples, real_augmentation, phase_augmentation)
                    pixels = torch.from_numpy(pixels)
                    labels = torch.from_numpy(labels).to(device)
                    mask = torch.from_numpy(mask).to(device)
                    output = model(pixels.to(device))
                    loss, ce = training_loss(output, labels, mask, samples[0].shape[1])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    (loss/accumulate).backward()
                    update_loss += float(loss.detach())/accumulate
                    wrong, cells = training_cell_counts(output.detach(), labels, mask, samples[0].shape[1])
                    metrics = torch.stack((ce.detach(), wrong, cells), dim=1).cpu().tolist()
                    for sample, (value, wrong_count, cell_count) in zip(samples, metrics):
                        source = sample.metadata["source"]
                        count = len(sample.members)
                        wrong_count, cell_count = int(wrong_count), int(cell_count)
                        interval["wrongCells"] += wrong_count
                        interval["dataCells"] += cell_count
                        interval["frameCounts"][str(count)] += 1
                        counters = [interval["sources"].setdefault(source, metric_counter()),
                                    interval["sizes"].setdefault(str(sample.shape[0]), metric_counter())]
                        if multires_data:
                            counters.append(interval["sizeSources"].setdefault(str(sample.shape[0]), {}).setdefault(
                                source, metric_counter()))
                        if source == "real":
                            name = sampler_groups[sample.metadata["sessionId"]]
                            counters.append(interval["groups"].setdefault(name, metric_counter()))
                        for counter in counters:
                            counter["lossSum"] += value
                            counter["samples"] += 1
                            counter["inputCaptures"] += count
                            counter["wrongCells"] += wrong_count
                            counter["dataCells"] += cell_count
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if multires_data:
                    learned_sizes.update(sample.shape[0] for samples in batches for sample in samples)
                interval["loss"] += update_loss
                interval["updates"] += 1
                for name in interval["sampling"]:
                    interval["sampling"][name] += sampler.last_counts.get(name, 0)
                completed = step
                update_progress(progress, interval)
                progress.update(1)
                if step % eval_every == 0 or step == steps:
                    evaluate(step, interval)
                    write_history()
                    update_progress(progress, interval, refresh=True)
                    interval = empty_interval()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        write_history()
        raise
    except Exception:
        stop_reason = "failed"
        write_history()
        raise
    return {"checkpoint": str(out/"best.pt"), "steps": completed, "requestedSteps": steps,
            "selectedStep": selected_step, "selectedExactRate": best[0], "stopReason": stop_reason,
            "elapsedSeconds": time.perf_counter()-started, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/neural-corrector/pilot"))
    parser.add_argument("--session-groups", type=Path, help="optional explicit acquisition groups from session_groups.py")
    parser.add_argument("--steps", type=int, help="maximum optimizer updates (default: 32,000)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4, help="microbatches per update; default effective batch is 16")
    parser.add_argument("--lr", type=float, help="default: 1e-4 fresh weights, 5e-5 initialized weights")
    parser.add_argument("--weight-decay", type=float, default=.01, help="AdamW decay (default: .01)")
    parser.add_argument("--frames", type=int, choices=FRAME_COUNTS,
                        metavar="1..64", help="input capacity: U-tube-2x-attn/multires use 8 phase-paired slots with 4–8 observations; other fresh profiles default to 1")
    parser.add_argument("--paired-targets", action="store_true",
                        help="pair real captures with synthetic examples of the same target; requires real-fraction .5")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=1000, help="validation interval (default: 1000 updates)")
    parser.add_argument("--validation-samples", type=int, default=0, help="0 evaluates all (default); positive values cap the diagnostic subset")
    parser.add_argument("--real-fraction", type=float, default=.5)
    parser.add_argument("--real-noise-std", type=float, default=.008,
                        help="real training noise standard deviation in RGB [0,1] units (default: .008)")
    parser.add_argument("--real-tint-stops", type=float, default=.06,
                        help="real training per-RGB gain range in stops, shared by the burst (default: .06)")
    parser.add_argument("--real-exposure-stops", type=float, default=.08,
                        help="real training exposure range in stops, shared by the burst (default: .08); set all three strengths to 0 to disable")
    parser.add_argument("--phase-flip-probability", type=float, default=.25,
                        help="Phase-grouped training: independent horizontal/vertical flip probability (default: .25)")
    parser.add_argument("--no-phase-permutations", action="store_true",
                        help="disable exact XOR phase-group augmentation")
    parser.add_argument("--model", choices=MODEL_PROFILES,
                        help=f"default: {DEFAULT_MODEL_PROFILE} fresh, otherwise checkpoint profile")
    parser.add_argument("--initialize", type=Path, help="initialize validated weights; --model multires also allows an explicit U-tube-2x-attn warm start; optimizer starts fresh")
    parser.add_argument("--node", default="node")
    parser.add_argument("--no-progress", action="store_true", help="hide the live bar; keep concise validation summaries")
    parser.add_argument("--show-goal", action="store_true", help="include the phone-to-PC 50%% recovery goal at validation events")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1–64")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    try:
        Dataset(list(dict.fromkeys(args.train+args.validation)))
        train = Dataset(args.train, "train")
        validation = Dataset(args.validation, "validation")
        options = vars(args).copy()
        for name in ("train", "validation", "threads"):
            options.pop(name)
        options["device"] = device
        options["session_groups"] = load_session_groups(args.session_groups) if args.session_groups else None
        options["learning_rate"] = options.pop("lr")
        options["model_profile"] = options.pop("model")
        result = run_training(train, validation, **options)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Training interrupted. Completed validation records remain in the run directory.\n")
    print(f"Finished {result['steps']}/{result['requestedSteps']} updates in {result['elapsedSeconds']:.1f}s. "
          f"Best step {result['selectedStep']}: selection exact {result['selectedExactRate']:.2%}.")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Training history: {Path(args.out).resolve()/'history.json'}")


if __name__ == "__main__":
    main()
