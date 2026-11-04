"""Bounded run summaries; capture-level diagnostics require explicit opt-in.

Detailed measurements are useful while aggregating exact recovery, but repeating
them at every validation step makes history grow with the dataset. This single
projection keeps reports proportional to the number of groups and input presets.
"""

from copy import deepcopy


METRIC_FIELDS = (
    "samples", "availableSamples", "observationCases", "uniqueCandidates",
    "uniqueAvailableCandidates", "framesPerInput", "frameCapacity",
    "actualFrameCounts", "outputLabelSpace", "inputCaptures",
    "uniqueInputCaptures", "availableInputCaptures", "dataCells", "loss",
    "cellErrorRate", "expectedCalibrationError",
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


def compact_report(report):
    """Return a non-mutating, idempotent summary without session/burst rows.

    The counts and checkpoint-selection objectives are copied exactly, not
    recomputed or rounded. Confusion matrices, calibration bins and capture IDs
    belong only in a separately requested detailed evaluation report.
    """
    if not isinstance(report, dict):
        raise ValueError("An evaluation report must be an object")
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
