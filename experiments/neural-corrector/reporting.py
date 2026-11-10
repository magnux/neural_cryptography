"""Bounded run summaries; capture-level diagnostics require explicit opt-in.

Detailed measurements are useful while aggregating exact recovery, but repeating
them at every validation step makes history grow with the dataset. This single
projection keeps reports proportional to the number of groups and input presets.
"""

from copy import deepcopy
import json
import math

from packet_replay import REPLAY_COUNTERS


TRAIN_COLUMNS = ("grid", "source", "group", "origin", "samples", "lossSum",
                 "wrongCells", "dataCells", "inputCaptures")
EVALUATION_COLUMNS = ("grid", "source", "group", "origin", "frames", "samples",
                      "lossSum", "wrongCells", "dataCells", "exactMatrices",
                      "inputCaptures", "classicalExactMatrices", *REPLAY_COUNTERS)


METRIC_FIELDS = (
    "samples", "availableSamples", "observationCases", "uniqueCandidates",
    "uniqueAvailableCandidates", "framesPerInput", "frameCapacity",
    "actualFrameCounts", "outputLabelSpace", "inputCaptures",
    "uniqueInputCaptures", "availableInputCaptures", "dataCells", "loss",
    "cellErrorRate", "sizeMacroCellErrorRate", "expectedCalibrationError",
)
GROUP_FIELDS = (
    "samples", "inputCaptures", "dataCells", "wrongCells", "loss",
    "cellErrorRate", "learnedExactFrames", "learnedExactRate",
    "classicalExactFrames", "classicalExactRate",
)
RECOVERY_FIELDS = (
    "framesPerInput", "frameCapacity", "actualFrameCounts", "outputLabelSpace",
    "inputCaptures", "uniqueInputCaptures", "observationBudget",
)
RECOVERY_METRICS = (
    "frames", "rawCRCFrames", "hardCRCFrames", "acceptedFrames", "exactFrames",
    "wrongAcceptedFrames", "exactRate",
)


def _fields(row, names):
    return {name: deepcopy(row[name]) for name in names if name in row}


def _metrics(report):
    result = _fields(report, METRIC_FIELDS)
    if "burstGrouping" in report:
        result["burstGrouping"] = _fields(report["burstGrouping"], (
            "inputCaptures", "usedCaptures", "discardedCaptures", "remainderCaptures",
            "unusedCapacityCaptures", "insufficientBursts", "completeBursts"))
    for kind in ("sources", "groups"):
        if kind in report:
            result[kind] = {name: _fields(row, GROUP_FIELDS)
                            for name, row in sorted(report[kind].items())}
    if "selectionScore" in report:
        result["selectionScore"] = _fields(report["selectionScore"],
            ("method", "groups", "frameCounts", "exactRate", "loss", "cellErrorRate"))
    if "recovery" in report:
        recovery = report["recovery"]
        result["recovery"] = None if recovery is None else _fields(recovery, RECOVERY_FIELDS)
        if recovery is not None:
            result["recovery"]["summary"] = {
                name: _fields(recovery["summary"][name], RECOVERY_METRICS)
                for name in ("classical", "learned") if name in recovery.get("summary", {})}
    if "inferenceMs" in report:
        timings = report["inferenceMs"]
        result["inferenceMs"] = _fields(timings, ("p50", "p95", "note"))
        if "byFrameCount" in timings:
            result["inferenceMs"]["byFrameCount"] = {
                key: _fields(row, ("p50", "p95"))
                for key, row in sorted(timings["byFrameCount"].items(), key=lambda item: int(item[0]))}
    return result


def _matrix_metrics(report):
    """Raw-symbol and separately labelled replay summaries, without case rows."""
    result = _fields(report, METRIC_FIELDS + ("wrongCells", "sizes", "rsReplay"))
    if "matrixAccuracy" in report:
        result["matrixAccuracy"] = _fields(report["matrixAccuracy"],
                                           ("matrices", "exactMatrices", "exactRate"))
    for kind in ("sources", "groups", "bySize", "byFrameCount"):
        if kind in report:
            key = (lambda item: int(item[0])) if kind in ("bySize", "byFrameCount") else (lambda item: item[0])
            result[kind] = {name: _matrix_metrics(row) for name, row in sorted(report[kind].items(), key=key)}
    if "burstGrouping" in report:
        result["burstGrouping"] = _fields(report["burstGrouping"], ("method", "seed", "framesPerInput",
            "frameCapacity", "actualFrameCounts", "inputCaptures", "usedCaptures", "discardedCaptures",
            "remainderCaptures", "unusedCapacityCaptures", "insufficientBursts", "completeBursts"))
    return result


def compact_report(report):
    """Return a non-mutating, idempotent summary without session/burst rows.

    The counts and checkpoint-selection objectives are copied exactly, not
    recomputed or rounded. Confusion matrices, calibration bins and capture IDs
    belong only in a separately requested detailed evaluation report.
    """
    if not isinstance(report, dict):
        raise ValueError("An evaluation report must be an object")
    if "matrixAccuracy" in report:
        result = _fields(report, ("format", "version", "modelProfile", "datasetFormat", "datasetVersion", "split",
                                  "checkpoint", "manifestSha256", "profiles", "pixelsPerCell", "metric", "note",
                                  "evaluationPopulation", "nativeAvailable", "primarySizes", "evaluatedSizes", "evaluatedCases", "rsReplayPolicy"))
        result.update(_matrix_metrics(report))
        result["reportDetail"] = "summary"
        if "selectionScore" in report:
            result["selectionScore"] = _fields(report["selectionScore"],
                ("method", "sizes", "groups", "frameCounts", "exactRate", "loss", "cellErrorRate",
                 "originScope", "nativeAvailable", "origins"))
        if "origins" in report:
            result["origins"] = {name: _matrix_metrics(row) for name, row in sorted(report["origins"].items())}
        if "syntheticValidation" in report:
            result["syntheticValidation"] = _matrix_metrics(report["syntheticValidation"])
        if "inferenceMs" in report:
            result["inferenceMs"] = _fields(report["inferenceMs"], ("p50", "p95", "note"))
        if "metricTable" in report:
            result["metricTable"] = deepcopy(report["metricTable"])
        return result
    result = _fields(report, ("format", "version", "modelProfile", "split",
                              "checkpoint", "manifestSha256", "profiles"))
    result["reportDetail"] = "summary"
    result.update(_metrics(report))
    if "selection" in report:
        selection = report["selection"]
        result["selection"] = _fields(selection, ("method", "seed", "maximum",
            "availableSessions", "selectedSessions", "omittedSessionCount", "note"))
        if "omittedSessions" in selection:
            result["selection"]["omittedSessionCount"] = len(selection["omittedSessions"])
    if "burstGrouping" in report:
        result["burstGrouping"] = _fields(report["burstGrouping"], (
            "method", "seed", "framesPerInput", "frameCapacity", "actualFrameCounts",
            "phaseCounts", "capturesPerPhase", "inputCaptures", "usedCaptures",
            "discardedCaptures", "remainderCaptures", "unusedCapacityCaptures",
            "insufficientBursts", "completeBursts", "uniqueCandidates", "note"))
    if "phoneToPCGoal" in report:
        goal = report["phoneToPCGoal"]
        result["phoneToPCGoal"] = _fields(goal, ("direction", "targetExactRate",
            "available", "exactRate", "achievedForAllFrameCounts"))
        result["phoneToPCGoal"]["byFrameCount"] = {
            key: _fields(row, ("available", "samples", "exactFrames", "exactRate", "achieved"))
            for key, row in sorted(goal.get("byFrameCount", {}).items(), key=lambda item: int(item[0]))}
    if "byFrameCount" in report:
        result["byFrameCount"] = {
            key: _metrics(row)
            for key, row in sorted(report["byFrameCount"].items(), key=lambda item: int(item[0]))}
    return result


def evaluation_table(leaves):
    """Canonical Neural-QR leaf counters, before any aggregate views are built.

    One row represents one grid/source/direction/origin/K combination. Its CE
    sum is over examples, while wrong/data cells retain the exact integer
    counts needed for cell-weighted CER. Classical recovery stays null because
    collection matrices contain no encoded packets. Separate replay counters
    describe diagnostic recovery after applying observed errors to test packets.
    """
    rows = []
    for key, counter in sorted(leaves.items()):
        grid, source, group, frames = key[:4]
        origin = key[4] if len(key) == 5 else None
        rows.append([grid, source, group, origin, frames, counter["samples"],
                     counter["lossSum"], counter["wrongCells"], counter["dataCells"],
                     counter["exactMatrices"], counter["inputCaptures"], None,
                     *(counter.get(field, 0) for field in REPLAY_COUNTERS)])
    return {"columns": list(EVALUATION_COLUMNS), "rows": rows}


def training_rows(counters):
    """Serialize each training grid/source/direction/origin counter once."""
    return [[*key, row["samples"], row["lossSum"], row["wrongCells"],
             row["dataCells"], row["inputCaptures"]]
            for key, row in sorted(counters.items(),
                                   key=lambda item: (item[0][0], *(
                                       "" if value is None else str(value) for value in item[0][1:])))]


def _packet_rows(report):
    """Flatten existing OBM group/K summaries without duplicating their totals.

    Those summaries pool geometry within each acquisition group, so grid and
    origin remain null. OBM CE sums are weighted by data cells; this explicit
    distinction is stored beside the table. Their packet exact counts are not
    reinterpreted as Neural-QR raw-cell exactness.
    """
    rows = []
    presets = report.get("byFrameCount") or {str(report.get("framesPerInput", 1)): report}
    for key, preset in sorted(presets.items(), key=lambda item: int(item[0])):
        groups = preset.get("groups") or preset.get("sources") or {"all": preset}
        for group, row in sorted(groups.items()):
            source = group[7:] if group.startswith("source:") else (
                group if group in ("real", "synthetic") else "real" if group != "all" else None)
            cells, samples = row.get("dataCells"), row.get("samples")
            loss = row.get("loss")
            rows.append([None, source, group, None, int(key), samples,
                         loss*cells if loss is not None and cells is not None else None,
                         row.get("wrongCells"), cells, row.get("learnedExactFrames"),
                         row.get("inputCaptures", samples*int(key) if samples is not None else None),
                         row.get("classicalExactFrames"), *(None for _ in REPLAY_COUNTERS)])
    return rows


def evaluation_step(report):
    """Small per-validation value for history; schema and notes live once above it."""
    matrix = "matrixAccuracy" in report
    headline = _fields(report, ("samples", "loss", "cellErrorRate", "sizeMacroCellErrorRate",
        "uniqueCandidates", "uniqueInputCaptures", "availableInputCaptures"))
    if matrix:
        headline["exactMatrices"] = report["matrixAccuracy"]["exactMatrices"]
        headline["exactRate"] = report["matrixAccuracy"]["exactRate"]
        if "rsReplay" in report:
            headline["rsReplay"] = deepcopy(report["rsReplay"])
        table = report.get("metricTable")
        if (not isinstance(table, dict) or table.get("columns") != list(EVALUATION_COLUMNS)
                or not isinstance(table.get("rows"), list)):
            raise ValueError("Neural-QR report requires canonical evaluation counters")
        rows = deepcopy(table["rows"])
    else:
        recovery = report.get("recovery")
        headline["recovery"] = deepcopy(recovery.get("summary")) if recovery else None
        rows = _packet_rows(report)
    selection = report.get("selectionScore")
    return {"headline": headline, "selection": None if selection is None else
            _fields(selection, ("exactRate", "loss", "cellErrorRate")),
            "coverage": _fields(report, ("evaluationPopulation", "nativeAvailable", "primarySizes",
                "evaluatedSizes", "evaluatedCases", "observationCases")),
            "lossWeight": "samples" if matrix else "dataCells", "rows": rows}


def evaluation_document(report):
    """Persist one current table format, leaving full-precision live metrics intact."""
    if report.get("format") == "neural-corrector-evaluation" and report.get("version") == 2:
        return deepcopy(report)
    metadata = _fields(report, ("modelProfile", "datasetFormat", "datasetVersion", "split",
        "checkpoint", "manifestSha256", "pixelsPerCell", "metric", "outputLabelSpace",
        "framesPerInput", "frameCapacity", "actualFrameCounts", "note", "rsReplayPolicy"))
    if "inferenceMs" in report:
        metadata["inferenceMs"] = _fields(report["inferenceMs"], ("p50", "p95", "note"))
    if "burstGrouping" in report:
        metadata["capturePolicy"] = _fields(report["burstGrouping"], ("method", "seed"))
    if report.get("selectionScore") is not None:
        metadata["selectionPolicy"] = _fields(report["selectionScore"], (
            "method", "sizes", "groups", "frameCounts", "originScope", "nativeAvailable", "origins"))
    if "metric" not in metadata:
        metadata["metric"] = "packet-recovery"
    return {"format": "neural-corrector-evaluation", "version": 2, "metadata": metadata,
            "tables": {"evaluation": list(EVALUATION_COLUMNS)}, **evaluation_step(report)}


def lean_json(value):
    """Round only serialized floats; keep exact counters and decision inputs intact."""
    def rounded(item):
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Report values must be finite")
            return float(format(item, ".8g"))
        if isinstance(item, dict):
            return {key: rounded(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [rounded(child) for child in item]
        return item
    return json.dumps(rounded(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
