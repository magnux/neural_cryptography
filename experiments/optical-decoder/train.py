#!/usr/bin/env python3
"""Train an optical classifier with a pilot or real-adaptation recipe.

Plain PyTorch, float32, AdamW. Adaptation balances acquisition groups, applies
mild real-only augmentation and mixes mean/hard-cell CE. Validation uses ordinary
CE and canonical exact recovery, never augmented inputs or the final test split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from adaptation import OnlineRealAugment, PlateauController, adaptation_loss
from checkpoint import load_model_checkpoint
from data import DATASET_FORMAT, Dataset, assert_disjoint, check_checkpoint_overlap, digest
from evaluate import evaluate_model
from model import MODEL_PROFILES, OpticalDecoder
from sampling import BalancedSampler, load_session_groups, validate_session_groups


def training_defaults(initialize=None, steps=None, learning_rate=None, *, recipe="pilot"):
    if recipe not in ("pilot", "adapt"):
        raise ValueError("Choose recipe pilot or adapt")
    default_lr = 1e-5 if recipe == "adapt" else (5e-5 if initialize is not None else 1e-4)
    return steps if steps is not None else 32_000, learning_rate if learning_rate is not None else default_lr


def save_checkpoint(path: Path, model, training: dict):
    value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
             "modelState": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
             "training": training}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_training(train: Dataset, validation: Dataset, out: Path, *, steps=None, batch_size=4,
                 accumulate=4, learning_rate=None, device="cpu", seed=7, eval_every=None,
                 validation_samples=None, real_fraction=.5, initialize=None, node="node", model_profile=None,
                 recipe=None, session_groups=None, weight_decay=None, hard_weight=None, hard_fraction=.25,
                 real_exposure_stops=None, real_white_balance_stops=None, real_noise_std=None,
                 lr_patience=None, stop_patience=None, min_lr=None, progress_delta=.001):
    has_real = any(s.metadata["source"] == "real" for s in train.samples)
    recipe = recipe or ("adapt" if initialize is not None and has_real else "pilot")
    steps, learning_rate = training_defaults(initialize, steps, learning_rate, recipe=recipe)
    adapting = recipe == "adapt"
    def setting(value, pilot, adapt):
        return (adapt if adapting else pilot) if value is None else value
    eval_every = setting(eval_every, 200, 1000)
    validation_samples = setting(validation_samples, 128, 0)
    weight_decay = setting(weight_decay, .01, .05)
    hard_weight = setting(hard_weight, 0., .1)
    real_exposure_stops = setting(real_exposure_stops, 0., .15)
    real_white_balance_stops = setting(real_white_balance_stops, 0., .1)
    real_noise_std = setting(real_noise_std, 0., .005)
    lr_patience = setting(lr_patience, 0, 4)
    stop_patience = setting(stop_patience, 0, 12)
    min_lr = min(2e-6, learning_rate) if min_lr is None else min_lr
    integers = ((steps, 1, 1_000_000), (batch_size, 1, 128), (accumulate, 1, 128),
                (seed, 0, 0xffffffff), (eval_every, 1, 1_000_000),
                (validation_samples, 0, 1_000_000_000))
    if (any(type(value) is not int or not low <= value <= high for value, low, high in integers)
            or any(isinstance(value, bool) for value in (learning_rate, weight_decay, hard_weight, hard_fraction))
            or not np.isfinite(learning_rate) or not 0 < learning_rate <= .1
            or not np.isfinite(weight_decay) or not 0 <= weight_decay <= 1
            or not np.isfinite(hard_weight) or not 0 <= hard_weight <= 1
            or not np.isfinite(hard_fraction) or not 0 < hard_fraction <= 1):
        raise ValueError("Invalid training budget, learning rate, weight decay, or loss settings")
    if any(s.metadata["split"] != "train" for s in train.samples):
        raise ValueError("Only train samples may be fitted")
    if any(s.metadata["split"] != "validation" for s in validation.samples):
        raise ValueError("Only validation samples may select checkpoints")
    assert_disjoint(train, validation)
    if adapting and (initialize is None or not has_real or session_groups is None):
        raise ValueError("The adapt recipe requires --initialize, real training data and --session-groups; create the sidecar with session_groups.py")
    if session_groups is not None:
        train_groups = validate_session_groups(train.samples, session_groups)
        validation_groups = validate_session_groups(validation.samples, session_groups)
        if adapting and (not validation_groups or train_groups != validation_groups):
            raise ValueError("Adaptation training and validation must represent the same real groups")
    # A pilot can use real data without direction weighting. Adaptation requires
    # the explicit sidecar; no camera direction is inferred from an opaque ID.
    sampler_groups = session_groups if session_groups is not None else {
        s.metadata["sessionId"]: "all-real" for s in train.samples if s.metadata["source"] == "real"}
    sampler = BalancedSampler(train.samples, sampler_groups, seed=seed, real_fraction=real_fraction)
    augment = OnlineRealAugment(seed=seed ^ 0x4155474d, exposure_stops=real_exposure_stops,
                                white_balance_stops=real_white_balance_stops, noise_std=real_noise_std)
    controller = PlateauController(learning_rate, min_lr=min_lr, lr_patience=lr_patience,
                                    stop_patience=stop_patience, min_delta=progress_delta)
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty run directory; existing runs are never overwritten")
    if model_profile is not None and model_profile not in MODEL_PROFILES:
        raise ValueError("Unknown decoder model profile")
    torch.manual_seed(seed)
    previous = None
    if initialize is None:
        model = OpticalDecoder(model_profile or "wide")
    else:
        model, previous = load_model_checkpoint(initialize)
        if model_profile is not None and model.profile != model_profile:
            raise ValueError("Requested --model differs from the initialized checkpoint profile; train that architecture from scratch")
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
    out.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    options = {"recipe": recipe, "requestedSteps": steps, "batchSize": batch_size,
               "gradientAccumulation": accumulate, "initialLearningRate": learning_rate,
               "weightDecay": weight_decay, "realFraction": real_fraction,
               "hardWeight": hard_weight, "hardFraction": hard_fraction,
               "realExposureStops": real_exposure_stops, "realWhiteBalanceStops": real_white_balance_stops,
               "realNoiseStd": real_noise_std, "evalEvery": eval_every, "validationSamples": validation_samples,
               "lrPatience": lr_patience, "stopPatience": stop_patience, "minLR": min_lr,
               "progressDelta": progress_delta, "sampling": "source-group-session-payload-phase"}
    provenance = {"seed": seed, "samples": len(train.samples), "batchSize": batch_size,
                  "gradientAccumulation": accumulate, "device": str(device), "realFraction": real_fraction,
                  "recipe": recipe, "selection": "validation-exact", "datasetFormat": DATASET_FORMAT,
                  "trainManifestSha256": train.manifest_hash, "validationManifestSha256": validation.manifest_hash,
                  "trainSessionIds": sorted(train_sessions), "trainPayloadIds": sorted(train_payloads),
                  "validationSessionIds": sorted(validation_sessions), "validationPayloadIds": sorted(validation_payloads),
                  "settings": options}
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
        report = evaluate_model(model, validation, device=device, max_samples=validation_samples,
                                node=node, work_dir=out, session_groups=session_groups)
        summary = report["recovery"]["summary"]
        if summary["learned"]["wrongAcceptedFrames"]:
            (out/"validation-failed.json").write_text(json.dumps(report, indent=2)+"\n")
            raise RuntimeError("Validation accepted incorrect payload bytes; inspect validation-failed.json")
        selection = report["selectionScore"]
        score = selection["exactRate"], -selection["loss"]
        # Step zero of a fresh pilot is diagnostic only, not a trained model.
        selected = score > best and (step > 0 or previous is not None)
        current_lr = optimizer.param_groups[0]["lr"]
        if selected:
            best, selected_step = score, step
            training = {**provenance, "steps": step, "selectionStep": step,
                        "baselineSelection": step == 0, "learningRate": current_lr,
                        "selectionAggregation": selection["method"],
                        "bestValidationExact": selection["exactRate"], "bestValidationLoss": selection["loss"]}
            save_checkpoint(out/"best.pt", model, training)
            (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
        schedule = controller.update(selection["exactRate"], selection["loss"])
        for group in optimizer.param_groups:
            group["lr"] = schedule["lr"]
        source_metrics = {name: {"samples": values[1], "meanCE": values[0]/values[1]}
                          for name, values in interval["sources"].items()}
        group_metrics = {name: {"samples": values[1], "meanCE": values[0]/values[1]}
                         for name, values in interval["groups"].items()}
        record = {"step": step, "trainLoss": interval["loss"]/interval["updates"] if interval["updates"] else None,
                  "trainSources": source_metrics, "trainGroups": group_metrics,
                  "sampling": interval["sampling"], "validationLoss": report["loss"],
                  "validationCellErrorRate": report["cellErrorRate"],
                  "validationExact": summary["learned"]["exactRate"],
                  "classicalExact": summary["classical"]["exactRate"],
                  "validationSelection": selection, "validationGroups": report["groups"],
                  "selected": selected, "learningRate": current_lr, "nextLearningRate": schedule["lr"],
                  "plateauChecks": schedule["badChecks"], "lrReduced": schedule["reduced"],
                  "elapsedSeconds": time.perf_counter()-started}
        history.append(record)
        print(json.dumps(record), flush=True)
        target = "baseline-validation.json" if step == 0 else "final-validation.json"
        (out/target).write_text(json.dumps(report, indent=2)+"\n")
        return schedule["shouldStop"]

    def empty_interval():
        return {"loss": 0., "updates": 0, "sources": {}, "groups": {},
                "sampling": {"repeatedBursts": 0, "repeatedSamples": 0}}

    def write_history():
        (out/"history.json").write_text(json.dumps({"format": "optical-decoder-training-run", "version": 1,
            "requestedSteps": steps, "completedSteps": completed, "selectedStep": selected_step,
            "stopReason": stop_reason, "device": str(device), "settings": options, "history": history,
            "note": "Validation selects checkpoints; untouched real test sessions establish generalization."}, indent=2)+"\n")

    evaluate(0, empty_interval())
    write_history()
    interval = empty_interval()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.
        batches = sampler.draw_update(batch_size, accumulate)
        for samples in batches:
            triples = [sample.tensors() for sample in samples]
            pixels = torch.from_numpy(np.stack([row[0] for row in triples]))
            real = [sample.metadata["source"] == "real" for sample in samples]
            pixels = augment(pixels, real)
            labels = torch.from_numpy(np.stack([row[1] for row in triples])).to(device)
            mask = torch.from_numpy(np.stack([row[2] for row in triples])).to(device)
            output = model(pixels.to(device))
            loss, ce = adaptation_loss(output, labels, mask, samples[0].shape[1], hard_weight, hard_fraction)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss/accumulate).backward()
            update_loss += float(loss.detach())/accumulate
            for sample, value in zip(samples, ce.detach().cpu().tolist()):
                source = sample.metadata["source"]
                counters = [interval["sources"].setdefault(source, [0., 0])]
                if source == "real":
                    name = sampler_groups[sample.metadata["sessionId"]]
                    counters.append(interval["groups"].setdefault(name, [0., 0]))
                for counter in counters:
                    counter[0] += value
                    counter[1] += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        interval["loss"] += update_loss
        interval["updates"] += 1
        for name in interval["sampling"]:
            interval["sampling"][name] += sampler.last_counts[name]
        completed = step
        if step % eval_every == 0 or step == steps:
            should_stop = evaluate(step, interval)
            if should_stop:
                stop_reason = "early-stopping"
            write_history()
            interval = empty_interval()
            if should_stop:
                break
    return {"checkpoint": str(out/"best.pt"), "steps": completed, "requestedSteps": steps,
            "selectedStep": selected_step, "selectedExactRate": best[0], "stopReason": stop_reason,
            "elapsedSeconds": time.perf_counter()-started, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/optical-decoder/pilot"))
    parser.add_argument("--recipe", choices=("pilot", "adapt"), help="default: adapt for initialized real training, otherwise pilot")
    parser.add_argument("--session-groups", type=Path, help="explicit receiver-session groups from session_groups.py; required for adapt")
    parser.add_argument("--steps", type=int, help="maximum optimizer updates (default: 32,000)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4, help="microbatches per update; default effective batch is 16")
    parser.add_argument("--lr", type=float, help="default: 1e-4 fresh pilot, 5e-5 initialized pilot, 1e-5 adapt")
    parser.add_argument("--weight-decay", type=float, help="AdamW decay (default: .01 pilot; .05 adapt)")
    parser.add_argument("--hard-weight", type=float, help="hard-cell CE mixture weight (default: 0 pilot; .1 adapt)")
    parser.add_argument("--hard-fraction", type=float, default=.25)
    parser.add_argument("--real-exposure-stops", type=float, help="online real-only maximum exposure (default: 0 pilot; .15 adapt)")
    parser.add_argument("--real-white-balance-stops", type=float, help="online real-only maximum channel gain (default: 0 pilot; .1 adapt)")
    parser.add_argument("--real-noise-std", type=float, help="online real-only RGB noise std (default: 0 pilot; .005 adapt)")
    parser.add_argument("--lr-patience", type=int, help="validation checks before halving LR (default: 0 pilot; 4 adapt); 0 disables")
    parser.add_argument("--stop-patience", type=int, help="checks before early stop (default: 0 pilot; 12 adapt); 0 disables")
    parser.add_argument("--min-lr", type=float, help="default: min(2e-6, initial learning rate)")
    parser.add_argument("--progress-delta", type=float, default=.001, help="relative CE improvement to reset patience when exact rate ties")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, help="default: 200 pilot; 1000 adapt")
    parser.add_argument("--validation-samples", type=int, help="0 evaluates all; default: 128 pilot; 0 adapt")
    parser.add_argument("--real-fraction", type=float, default=.5)
    parser.add_argument("--model", choices=MODEL_PROFILES, help="default: wide fresh, otherwise checkpoint profile")
    parser.add_argument("--initialize", type=Path, help="initialize validated weights; optimizer starts fresh")
    parser.add_argument("--node", default="node")
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
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
