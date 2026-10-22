#!/usr/bin/env python3
"""Train the small optical classifier on session-disjoint M2 datasets.

Single-frame, plain PyTorch, float32, AdamW. Select the checkpoint by held-out
canonical exact-frame recovery, breaking ties by data-cell cross-entropy.
The final test split is never used for fitting or checkpoint selection.
"""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from checkpoint import load_checkpoint
from data import DATASET_FORMAT, Dataset, assert_disjoint, check_checkpoint_overlap, digest
from evaluate import evaluate_model, masked_loss
from model import OpticalDecoder


def choose_batch(groups, rng, batch_size: int, real_fraction: float):
    """One palette and tensor shape per microbatch; source mixture is explicit."""
    shape = rng.choice(sorted(groups))
    values = groups[shape]
    real = [sample for sample in values if sample.metadata["source"] == "real"]
    synthetic = [sample for sample in values if sample.metadata["source"] == "synthetic"]
    chosen = []
    for _ in range(batch_size):
        pool = real if real and (not synthetic or rng.random() < real_fraction) else synthetic
        chosen.append(rng.choice(pool))
    triples = [sample.tensors() for sample in chosen]
    return (torch.from_numpy(np.stack([x[0] for x in triples])),
            torch.from_numpy(np.stack([x[1] for x in triples])),
            torch.from_numpy(np.stack([x[2] for x in triples])), shape[1])


def save_checkpoint(path: Path, model, training: dict):
    value = {"format": "optical-decoder-checkpoint", "version": 1, "status": "trained",
             "modelState": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
             "training": training}
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_training(train: Dataset, validation: Dataset, out: Path, *, steps=2000, batch_size=4,
                 accumulate=4, learning_rate=1e-3, device="cpu", seed=7, eval_every=200,
                 validation_samples=128, real_fraction=.5, initialize=None, node="node"):
    if (not 1 <= steps <= 1_000_000 or not 1 <= batch_size <= 128 or not 1 <= accumulate <= 128
            or not 0 < learning_rate <= .1 or not 0 <= real_fraction <= 1
            or not 0 <= seed <= 0xffffffff or eval_every < 1 or validation_samples < 1):
        raise ValueError("Invalid training budget, seed, learning rate, or source mixture")
    if any(s.metadata["split"] != "train" for s in train.samples):
        raise ValueError("Only train samples may be fitted")
    if any(s.metadata["split"] != "validation" for s in validation.samples):
        raise ValueError("Only validation samples may select checkpoints")
    assert_disjoint(train, validation)
    out = Path(out).resolve()
    if (out/"best.pt").exists():
        raise ValueError("Output already contains best.pt; choose a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = OpticalDecoder()
    train_sessions, train_payloads = set(train.session_ids), set(train.payload_ids)
    validation_sessions, validation_payloads = set(validation.session_ids), set(validation.payload_ids)
    if initialize is not None:
        previous = load_checkpoint(initialize, model)
        check_checkpoint_overlap(previous, validation, "validation")
        # A previous model-selection session cannot become training material.
        previous_training = previous["training"]
        if (train.session_ids & set(previous_training["validationSessionIds"])
                or train.payload_ids & set(previous_training["validationPayloadIds"])):
            raise ValueError("Fine-tuning data overlaps previous checkpoint-selection data")
        train_sessions.update(previous_training["trainSessionIds"])
        train_payloads.update(previous_training["trainPayloadIds"])
        validation_sessions.update(previous_training["validationSessionIds"])
        validation_payloads.update(previous_training["validationPayloadIds"])
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    groups = train.groups()
    history, best = [], (-1., -float("inf"))
    started = time.perf_counter()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for _ in range(accumulate):
            pixels, labels, mask, colors = choose_batch(groups, rng, batch_size, real_fraction)
            output = model(pixels.to(device))
            loss = masked_loss(output, labels.to(device), mask.to(device), colors)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss/accumulate).backward()
            total_loss += float(loss.detach())/accumulate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            report = evaluate_model(model, validation, device=device, max_samples=validation_samples,
                                    node=node, work_dir=out)
            summary = report["recovery"]["summary"]
            if summary["learned"]["wrongAcceptedFrames"]:
                (out/"validation-failed.json").write_text(json.dumps(report, indent=2)+"\n")
                raise RuntimeError("Validation accepted incorrect payload bytes; inspect validation-failed.json before continuing")
            exact_rate = summary["learned"]["exactRate"]
            score = (exact_rate, -report["loss"])
            selected = score > best
            record = {"step": step, "trainLoss": total_loss, "validationLoss": report["loss"],
                      "validationCellErrorRate": report["cellErrorRate"],
                      "validationExact": exact_rate, "classicalExact": summary["classical"]["exactRate"],
                      "selected": selected, "elapsedSeconds": time.perf_counter()-started}
            history.append(record)
            print(json.dumps(record), flush=True)
            if selected:
                best = score
                training = {"steps": step, "seed": seed, "samples": len(train.samples),
                    "batchSize": batch_size, "gradientAccumulation": accumulate,
                    "learningRate": learning_rate, "device": str(device), "realFraction": real_fraction,
                    "selection": "validation-exact", "selectionStep": step,
                    "bestValidationExact": exact_rate, "bestValidationLoss": report["loss"],
                    "datasetFormat": DATASET_FORMAT, "trainManifestSha256": train.manifest_hash,
                    "validationManifestSha256": validation.manifest_hash,
                    "trainSessionIds": sorted(train_sessions), "trainPayloadIds": sorted(train_payloads),
                    "validationSessionIds": sorted(validation_sessions), "validationPayloadIds": sorted(validation_payloads)}
                if initialize is not None:
                    training["initialCheckpointSha256"] = digest(Path(initialize))
                save_checkpoint(out/"best.pt", model, training)
                (out/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
            (out/"history.json").write_text(json.dumps({"format": "optical-decoder-training-run", "version": 1,
                "requestedSteps": steps, "device": str(device), "history": history,
                "note": "Synthetic validation is not evidence of real-device generalization."}, indent=2)+"\n")
    return {"checkpoint": str(out/"best.pt"), "steps": steps, "selectedExactRate": best[0],
            "elapsedSeconds": time.perf_counter()-started, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/optical-decoder/pilot"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4, help="microbatches per update; default effective batch is 16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--real-fraction", type=float, default=.5)
    parser.add_argument("--initialize", type=Path, help="fine-tune a validated checkpoint; optimizer starts fresh")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1–64")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    try:
        # Load all manifests together once so even filtered-out groups cannot
        # silently carry conflicting split labels across input directories.
        Dataset(list(dict.fromkeys(args.train+args.validation)))
        train = Dataset(args.train, "train")
        validation = Dataset(args.validation, "validation")
        result = run_training(train, validation, args.out, steps=args.steps, batch_size=args.batch_size,
            accumulate=args.accumulate, learning_rate=args.lr, device=device, seed=args.seed,
            eval_every=args.eval_every, validation_samples=args.validation_samples,
            real_fraction=args.real_fraction, initialize=args.initialize, node=args.node)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
