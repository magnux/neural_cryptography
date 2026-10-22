"""Current optical checkpoint validation shared by training, evaluation, and export."""

from pathlib import Path
import json
import pickle
import zipfile

import numpy as np
import torch

from model import OpticalDecoder


MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


def training_provenance(training: dict) -> dict:
    """Publish only bounded, understood metadata; never propagate local paths."""
    if not isinstance(training, dict):
        raise ValueError("Checkpoint training metadata must be an object")
    try:
        metadata = json.dumps(training, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint training metadata must contain finite JSON values") from error
    if len(metadata.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Checkpoint training metadata exceeds 1 MiB")
    result = {}
    for key, minimum, maximum in (("steps", 1, 10_000_000), ("seed", 0, 0xFFFFFFFF),
                                   ("samples", 1, 1_000_000_000), ("batchSize", 1, 65536),
                                   ("gradientAccumulation", 1, 65536),
                                   ("selectionStep", 1, 10_000_000)):
        if key not in training:
            if key in ("steps", "seed"):
                raise ValueError(f"Checkpoint training metadata requires {key}")
            continue
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid checkpoint training {key}")
        result[key] = value
    for key in ("trainManifestSha256", "validationManifestSha256", "initialCheckpointSha256"):
        if key in training:
            value = training[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = value
    if "selection" in training:
        if training["selection"] not in ("last-step", "validation-exact"):
            raise ValueError("Unsupported checkpoint selection method")
        result["selection"] = training["selection"]
    if "learningRate" in training:
        value = training["learningRate"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Invalid checkpoint learning rate")
        result["learningRate"] = float(value)
    if "datasetFormat" in training:
        if training["datasetFormat"] != "optical-decoder-dataset":
            raise ValueError("Unsupported checkpoint dataset format")
        result["datasetFormat"] = training["datasetFormat"]
    for key in ("bestValidationExact", "bestValidationLoss"):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = float(value)
    if "realFraction" in training:
        value = training["realFraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid checkpoint real data fraction")
        result["realFraction"] = float(value)
    return result


def validate_split_provenance(training: dict) -> None:
    """Keep full split identities locally so evaluation can reject data leakage."""
    for key in ("trainSessionIds", "trainPayloadIds", "validationSessionIds", "validationPayloadIds"):
        values = training.get(key)
        if not isinstance(values, list) or len(values) > 100_000 or any(
                not isinstance(value, str) or not 1 <= len(value) <= 128 for value in values):
            raise ValueError(f"Checkpoint requires bounded {key} identities")
        if len(values) != len(set(values)):
            raise ValueError(f"Checkpoint contains duplicate {key} identities")
    if set(training["trainSessionIds"]) & set(training["validationSessionIds"]):
        raise ValueError("Checkpoint training and validation sessions overlap")
    if set(training["trainPayloadIds"]) & set(training["validationPayloadIds"]):
        raise ValueError("Checkpoint training and validation payloads overlap")


def load_checkpoint(path: Path, model: OpticalDecoder | None = None) -> dict:
    """Load one current plain state checkpoint with strict dtype/shape checks."""
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint must be between 1 byte and 16 MiB")
    if not zipfile.is_zipfile(path):
        raise ValueError("Checkpoint must use the current torch.save ZIP serialization")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Expanded checkpoint exceeds 16 MiB")
    # weights_only rejects executable/custom pickle classes; map_location never
    # allocates CUDA tensors merely because the training checkpoint used CUDA.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("Checkpoint contains unsupported or malformed serialized values") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "optical-decoder-checkpoint" or \
            type(checkpoint.get("version")) is not int or checkpoint["version"] != 1 or checkpoint.get("status") != "trained":
        raise ValueError("Unsupported decoder checkpoint; use the current training command")
    state = checkpoint.get("modelState")
    model = OpticalDecoder() if model is None else model
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint model parameters do not match the decoder")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided or \
                tuple(value.shape) != tuple(expected[name].shape) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    training_provenance(checkpoint.get("training"))
    validate_split_provenance(checkpoint["training"])
    model.load_state_dict(state, strict=True)
    return checkpoint
