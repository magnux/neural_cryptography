"""Current optical checkpoint validation shared by training, evaluation, and export."""

from pathlib import Path
import json
import pickle
import zipfile

import numpy as np
import torch

from contract import FRAME_COUNTS, NETWORK_INPUT_MIN, NETWORK_INPUT_MAX
from model import (MODEL_PROFILES, MULTIRES_SIZES, ANCHOR_METADATA, BOTTLENECK_METADATA,
                   W_TUBE_PROFILES, wtube_bottleneck_metadata,
                   NeuralCorrector, profile_from_state, frames_from_state, project_anchor)


MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 256 * 1024 * 1024


def training_provenance(training: dict) -> dict:
    """Publish only bounded, understood metadata; never propagate local paths."""
    if not isinstance(training, dict):
        raise ValueError("Checkpoint training metadata must be an object")
    try:
        metadata = json.dumps(training, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint training metadata must contain finite JSON values") from error
    if len(metadata.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("Checkpoint training metadata exceeds 256 MiB")
    result = {}
    baseline = training.get("baselineSelection", False)
    if "baselineSelection" in training:
        if not isinstance(baseline, bool):
            raise ValueError("Invalid checkpoint baselineSelection")
        result["baselineSelection"] = baseline
    if baseline and (training.get("steps") != 0 or training.get("selectionStep") != 0
                     or "initialCheckpointSha256" not in training):
        raise ValueError("A baseline selection requires zero steps/selectionStep and an initial checkpoint hash")
    minimum_step = 0 if baseline else 1
    for key, minimum, maximum in (("steps", minimum_step, 10_000_000), ("seed", 0, 0xFFFFFFFF),
                                   ("samples", 1, 1_000_000_000), ("batchSize", 1, 65536),
                                   ("gradientAccumulation", 1, 65536),
                                   ("selectionStep", minimum_step, 10_000_000)):
        if key not in training:
            if key in ("steps", "seed"):
                raise ValueError(f"Checkpoint training metadata requires {key}")
            continue
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid checkpoint training {key}")
        result[key] = value
    for key in ("trainManifestSha256", "validationManifestSha256", "initialCheckpointSha256", "sessionGroupsSha256"):
        if key in training:
            value = training[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = value
    if "framesPerInput" in training:
        value = training["framesPerInput"]
        if type(value) is not int or value not in FRAME_COUNTS:
            raise ValueError("Invalid checkpoint framesPerInput")
        result["framesPerInput"] = value
    if "minFramesPerInput" in training:
        value = training["minFramesPerInput"]
        if type(value) is not int or value not in FRAME_COUNTS:
            raise ValueError("Invalid checkpoint minFramesPerInput")
        result["minFramesPerInput"] = value
    if "frameCounts" in training:
        values = training["frameCounts"]
        if (not isinstance(values, list) or not values or
                any(type(value) is not int or value not in FRAME_COUNTS for value in values) or
                values != sorted(set(values))):
            raise ValueError("Invalid checkpoint frameCounts")
        result["frameCounts"] = values
    for key, choices in (("inputLayout", ("stacked-rgb", "phase-paired-rgb-presence-planes")),
                         ("outputLabelSpace", ("physical", "logical")),
                         ("frameCountPolicy", ("fixed", "uniform-4-8"))):
        if key in training:
            if training[key] not in choices:
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = training[key]
    if "phaseMasks" in training:
        values = training["phaseMasks"]
        if (not isinstance(values, list) or any(type(value) is not int for value in values)
                or values not in ([], [0, 15, 7, 8])):
            raise ValueError("Invalid checkpoint phaseMasks")
        result["phaseMasks"] = values
    if "modelProfile" in training:
        if training["modelProfile"] not in MODEL_PROFILES:
            raise ValueError("Invalid checkpoint modelProfile")
        result["modelProfile"] = training["modelProfile"]
    bottleneck = (wtube_bottleneck_metadata(training["modelProfile"])
                  if training.get("modelProfile") in W_TUBE_PROFILES else BOTTLENECK_METADATA)
    for key, expected in (("anchor", ANCHOR_METADATA), ("bottleneck", bottleneck)):
        if key in training:
            value = training[key]
            integer_key = "maxModes" if key == "anchor" else "layersPerBlock"
            if (not isinstance(value, dict) or value != expected
                    or type(value.get(integer_key)) is not int):
                raise ValueError(f"Invalid checkpoint {key}: use the declared model architecture")
            if key == "bottleneck" and training.get("modelProfile") in W_TUBE_PROFILES:
                if (type(value.get("stages")) is not int
                        or any(type(scale) is not int for scale in value.get("scales", []))):
                    raise ValueError("Invalid W-tube bottleneck geometry metadata")
            result[key] = json.loads(json.dumps(value))
    for key in ("supportedSizes", "trainedSizes"):
        if key in training:
            values = training[key]
            if (not isinstance(values, list) or not values or
                    any(type(size) is not int or size not in MULTIRES_SIZES for size in values)
                    or values != sorted(set(values))):
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = values
    if "pixelsPerCell" in training:
        value = training["pixelsPerCell"]
        if type(value) is not int or value not in (2, 4):
            raise ValueError("Invalid checkpoint pixelsPerCell")
        result["pixelsPerCell"] = value
    if "networkInputClip" in training:
        value = training["networkInputClip"]
        if (not isinstance(value, list) or len(value) != 2
                or any(type(bound) not in (int, float) for bound in value)
                or value != [NETWORK_INPUT_MIN, NETWORK_INPUT_MAX]):
            raise ValueError("Invalid checkpoint networkInputClip")
        result["networkInputClip"] = value
    if "pairedTargets" in training:
        if not isinstance(training["pairedTargets"], bool):
            raise ValueError("Invalid checkpoint pairedTargets")
        result["pairedTargets"] = training["pairedTargets"]
    for key, choices in (("selectionAggregation", ("global", "group-balanced-real",
                            "frame-count-balanced", "frame-count-group-balanced-real",
                            "size-group-frame-balanced-real")),):
        if key in training:
            if training[key] not in choices:
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = training[key]
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
        if training["datasetFormat"] not in ("optical-decoder-dataset", "neural-qr-dataset"):
            raise ValueError("Unsupported checkpoint dataset format")
        result["datasetFormat"] = training["datasetFormat"]
    if "datasetVersion" in training:
        if type(training["datasetVersion"]) is not int or training["datasetVersion"] not in (1, 2):
            raise ValueError("Unsupported checkpoint dataset version")
        result["datasetVersion"] = training["datasetVersion"]
    for key in ("bestValidationExact", "bestValidationLoss"):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = float(value)
    for key in ("realFraction", "nativeRealFraction"):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid checkpoint {key}")
            result[key] = float(value)
    return result


def validate_split_provenance(training: dict) -> None:
    """Keep full split identities locally so evaluation can reject data leakage."""
    for key in ("trainSessionIds", "trainPayloadIds", "validationSessionIds", "validationPayloadIds"):
        values = training.get(key)
        if not isinstance(values, list) or any(
                not isinstance(value, str) or not 1 <= len(value) <= 128 for value in values):
            raise ValueError(f"Checkpoint requires bounded {key} identities")
        if len(values) != len(set(values)):
            raise ValueError(f"Checkpoint contains duplicate {key} identities")
    if set(training["trainSessionIds"]) & set(training["validationSessionIds"]):
        raise ValueError("Checkpoint training and validation sessions overlap")
    if set(training["trainPayloadIds"]) & set(training["validationPayloadIds"]):
        raise ValueError("Checkpoint training and validation payloads overlap")


def load_checkpoint(path: Path, model: NeuralCorrector | None = None) -> dict:
    """Load one current plain state checkpoint with strict dtype/shape checks."""
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint must be between 1 byte and 512 MiB")
    if not zipfile.is_zipfile(path):
        raise ValueError("Checkpoint must use the current torch.save ZIP serialization")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Expanded checkpoint exceeds 512 MiB")
    # weights_only rejects executable/custom pickle classes; map_location never
    # allocates CUDA tensors merely because the training checkpoint used CUDA.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError("Checkpoint contains unsupported or malformed serialized values") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "optical-decoder-checkpoint" or \
            type(checkpoint.get("version")) is not int or checkpoint["version"] != 1 or checkpoint.get("status") != "trained":
        raise ValueError("Unsupported corrector checkpoint; use the current training command")
    state = checkpoint.get("modelState")
    model = NeuralCorrector(profile_from_state(state), frames=frames_from_state(state)) if model is None else model
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint model parameters do not match the corrector")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided or \
                tuple(value.shape) != tuple(expected[name].shape) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    training_provenance(checkpoint.get("training"))
    if checkpoint["training"].get("framesPerInput", model.frames) != model.frames:
        raise ValueError("Checkpoint frame metadata disagrees with model parameters")
    if checkpoint["training"].get("modelProfile", model.profile) != model.profile:
        raise ValueError("Checkpoint profile metadata disagrees with model parameters")
    frame_metadata = {"inputLayout": model.input_layout,
                      "outputLabelSpace": model.output_label_space,
                      "phaseMasks": list(model.phase_masks),
                      "minFramesPerInput": model.min_frames,
                      "frameCounts": list(range(model.min_frames, model.frames+1)),
                      "frameCountPolicy": "uniform-4-8" if model.variable_frames else "fixed"}
    for name, expected_value in frame_metadata.items():
        training = checkpoint["training"]
        if model.variable_frames and name not in training:
            raise ValueError(f"{model.profile} checkpoint requires {name}")
        if name in training and training[name] != expected_value:
            raise ValueError(f"Checkpoint {name} disagrees with model parameters")
    if model.variable_frames and checkpoint["training"].get("framesPerInput") != 8:
        raise ValueError(f"{model.profile} checkpoint requires eight-slot frame metadata")
    if model.profile in ("multires", *W_TUBE_PROFILES):
        geometry = {"modelProfile": model.profile, "datasetFormat": "neural-qr-dataset",
                    "pixelsPerCell": model.pixels_per_cell, "supportedSizes": list(model.supported_sizes),
                    "anchor": model.anchor_metadata, "bottleneck": model.bottleneck_metadata}
        if model.profile in W_TUBE_PROFILES:
            geometry["datasetVersion"] = 2
        for name, expected_value in geometry.items():
            if checkpoint["training"].get(name) != expected_value:
                raise ValueError(f"{model.profile} checkpoint requires {name}={expected_value}")
        if not checkpoint["training"].get("trainedSizes"):
            raise ValueError(f"{model.profile} checkpoint requires trainedSizes to identify learned anchors")
        if set(checkpoint["training"]["trainedSizes"])-set(model.supported_sizes):
            raise ValueError("Checkpoint trainedSizes contains an unsupported model grid")
    elif checkpoint["training"].get("datasetFormat", "optical-decoder-dataset") != "optical-decoder-dataset":
        raise ValueError("U-tube checkpoints require the established optical-decoder-dataset geometry")
    validate_split_provenance(checkpoint["training"])
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_model_checkpoint(path: Path) -> tuple[NeuralCorrector, dict]:
    """Instantiate the exact supported profile represented by a validated state."""
    checkpoint = load_checkpoint(path)
    model = NeuralCorrector(profile_from_state(checkpoint["modelState"]),
                           frames=frames_from_state(checkpoint["modelState"]))
    model.load_state_dict(checkpoint["modelState"], strict=True)
    return model, checkpoint


def initialize_multires(source: NeuralCorrector) -> tuple[NeuralCorrector, dict]:
    """Explicit partial warm start from the successful four-pixel attention U.

    Shape-compatible shared blocks transfer exactly. The G64 anchor is projected
    onto the retained low-frequency basis. The second and third convolution/
    attention segments and five other anchors start fresh. Optimizer/step state
    is deliberately not inherited: this changes the
    computation and input cell density. Exact multires initialization uses
    load_model_checkpoint instead, which never tolerates missing tensors.
    """
    if source.profile != "U-tube-2x-attn":
        raise ValueError("multires warm start requires an explicit U-tube-2x-attn source")
    parameter = next(source.parameters())
    model = NeuralCorrector("multires").to(device=parameter.device, dtype=parameter.dtype)
    source_state = source.state_dict()
    target_state = model.state_dict()
    copied = []
    source_keys = set()
    for name, value in target_state.items():
        candidate = source_state.get(name)
        if candidate is not None and candidate.shape == value.shape:
            target_state[name] = candidate.detach().clone()
            copied.append(name)
            source_keys.add(name)
    projected = ["anchor_coefficients.64"]
    target_state[projected[0]] = project_anchor(source.position_anchor.detach()).clone()
    source_keys.add("position_anchor")
    model.load_state_dict(target_state, strict=True)
    model.train(source.training)
    report = {"mode": "warm-start", "sourceProfile": source.profile,
              "targetProfile": model.profile, "copied": sorted(copied),
              "projected": projected,
              "fresh": sorted(set(target_state)-set(copied)-set(projected)),
              "discarded": sorted(set(source_state)-source_keys),
              "transferredAnchors": [64]}
    return model, report
