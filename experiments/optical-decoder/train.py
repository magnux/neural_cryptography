#!/usr/bin/env python3
"""Train an optical classifier with ordinary masked cross-entropy.

Plain PyTorch, float32, AdamW. Optional target pairing places real captures and
synthetic renders of the same matrix in each optimizer update. Validation uses
held-out sessions and canonical exact recovery, never the final test split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from torch.nn import functional as F
from checkpoint import load_model_checkpoint
from data import DATASET_FORMAT, Dataset, assert_disjoint, check_checkpoint_overlap, digest
from evaluate import evaluate_model
from model import MODEL_PROFILES, OpticalDecoder
from sampling import BalancedSampler, load_session_groups, validate_session_groups


def training_defaults(initialize=None, steps=None, learning_rate=None):
    default_lr = 5e-5 if initialize is not None else 1e-4
    return steps if steps is not None else 32_000, learning_rate if learning_rate is not None else default_lr


def training_loss(logits, labels, mask, colors):
    """Mean data-cell CE per frame, then equal weight for every batch member."""
    counts = mask.flatten(1).sum(1)
    if colors not in (4, 16) or not torch.all(counts > 0):
        raise ValueError("A supported palette and nonempty data-cell mask per frame are required")
    cells = F.cross_entropy(logits[:, :colors], labels, reduction="none")
    per_frame = (cells*mask).flatten(1).sum(1)/counts
    return per_frame.mean(), per_frame


def save_checkpoint(path: Path, model, training: dict):
    value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
             "modelState": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
             "training": training}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_training(train: Dataset, validation: Dataset, out: Path, *, steps=None, batch_size=4,
                 accumulate=4, learning_rate=None, device="cpu", seed=7, eval_every=1000,
                 validation_samples=0, real_fraction=.5, initialize=None, node="node", model_profile=None,
                 session_groups=None, weight_decay=.01, paired_targets=False):
    steps, learning_rate = training_defaults(initialize, steps, learning_rate)
    integers = ((steps, 1, 1_000_000), (batch_size, 1, 128), (accumulate, 1, 128),
                (seed, 0, 0xffffffff), (eval_every, 1, 1_000_000),
                (validation_samples, 0, 1_000_000_000))
    if (any(type(value) is not int or not low <= value <= high for value, low, high in integers)
            or any(isinstance(value, bool) for value in (learning_rate, weight_decay))
            or not np.isfinite(learning_rate) or not 0 < learning_rate <= .1
            or not np.isfinite(weight_decay) or not 0 <= weight_decay <= 1
            or not isinstance(paired_targets, bool)):
        raise ValueError("Invalid training budget, learning rate, weight decay, or pairing setting")
    if any(s.metadata["split"] != "train" for s in train.samples):
        raise ValueError("Only train samples may be fitted")
    if any(s.metadata["split"] != "validation" for s in validation.samples):
        raise ValueError("Only validation samples may select checkpoints")
    assert_disjoint(train, validation)
    if session_groups is not None:
        validate_session_groups(train.samples, session_groups)
        validate_session_groups(validation.samples, session_groups)
    # Optional acquisition directions are explicit; never infer them from IDs.
    sampler_groups = session_groups if session_groups is not None else {
        s.metadata["sessionId"]: "all-real" for s in train.samples if s.metadata["source"] == "real"}
    sampler = BalancedSampler(train.samples, sampler_groups, seed=seed, real_fraction=real_fraction,
                              paired_targets=paired_targets)
    if paired_targets and batch_size*accumulate % 2:
        raise ValueError("Paired targets require an even effective batch size")
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
    options = {"requestedSteps": steps, "batchSize": batch_size,
               "gradientAccumulation": accumulate, "initialLearningRate": learning_rate,
               "weightDecay": weight_decay, "realFraction": real_fraction,
               "pairedTargets": paired_targets, "loss": "mean-masked-cross-entropy",
               "evalEvery": eval_every, "validationSamples": validation_samples,
               "sampling": "paired-targets" if paired_targets else "source-group-session-payload-phase"}
    provenance = {"seed": seed, "samples": len(train.samples), "batchSize": batch_size,
                  "gradientAccumulation": accumulate, "device": str(device), "realFraction": real_fraction,
                  "pairedTargets": paired_targets, "selection": "validation-exact", "datasetFormat": DATASET_FORMAT,
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
        # Step zero with fresh weights is diagnostic only, not a trained model.
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
                  "selected": selected, "learningRate": current_lr,
                  "elapsedSeconds": time.perf_counter()-started}
        history.append(record)
        print(json.dumps(record), flush=True)
        target = "baseline-validation.json" if step == 0 else "final-validation.json"
        (out/target).write_text(json.dumps(report, indent=2)+"\n")

    def empty_interval():
        return {"loss": 0., "updates": 0, "sources": {}, "groups": {},
                "sampling": {"repeatedBursts": 0, "repeatedSamples": 0, "pairedTargets": 0}}

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
            labels = torch.from_numpy(np.stack([row[1] for row in triples])).to(device)
            mask = torch.from_numpy(np.stack([row[2] for row in triples])).to(device)
            output = model(pixels.to(device))
            loss, ce = training_loss(output, labels, mask, samples[0].shape[1])
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
            interval["sampling"][name] += sampler.last_counts.get(name, 0)
        completed = step
        if step % eval_every == 0 or step == steps:
            evaluate(step, interval)
            write_history()
            interval = empty_interval()
    return {"checkpoint": str(out/"best.pt"), "steps": completed, "requestedSteps": steps,
            "selectedStep": selected_step, "selectedExactRate": best[0], "stopReason": stop_reason,
            "elapsedSeconds": time.perf_counter()-started, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/optical-decoder/pilot"))
    parser.add_argument("--session-groups", type=Path, help="optional explicit acquisition groups from session_groups.py")
    parser.add_argument("--steps", type=int, help="maximum optimizer updates (default: 32,000)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4, help="microbatches per update; default effective batch is 16")
    parser.add_argument("--lr", type=float, help="default: 1e-4 fresh weights, 5e-5 initialized weights")
    parser.add_argument("--weight-decay", type=float, default=.01, help="AdamW decay (default: .01)")
    parser.add_argument("--paired-targets", action="store_true",
                        help="pair real captures with synthetic examples of the same target; requires real-fraction .5")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=1000, help="validation interval (default: 1000 updates)")
    parser.add_argument("--validation-samples", type=int, default=0, help="0 evaluates all (default); positive values cap the diagnostic subset")
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
