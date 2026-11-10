"""Recorded-error RS/CRC replay, separate from raw optical matrix accuracy.

Raw Neural-QR collection targets contain no packet redundancy. After inference,
their observed symbol errors are replayed on an encoded test packet. Only that
packet reaches the canonical JS decoder; expected bytes never repair predictions.
Requests and responses are spooled as JSONL, with one Node process per evaluation.
"""

import base64
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPLAY_COUNTERS = {
    "rsReplayCases": "cases",
    "rsEligibleCases": "eligibleCases",
    "rsRawCRCValidCases": "rawCRCValidCases",
    "rsDecodedCases": "rsDecodedCases",
    "rsCRCAcceptedCases": "crcAcceptedCases",
    "rsExactCases": "exactCases",
    "rsWrongAcceptedCases": "wrongAcceptedCases",
    "rsPayloadBytes": "payloadBytes",
    "rsWireBytes": "wireBytes",
    "rsParityBytes": "parityBytes",
    "rsBlocks": "blocks",
}


def validate_rs_parity(value):
    if type(value) is not int or value != 0 and (value < 2 or value > 64 or value % 2):
        raise ValueError("rs-parity must be 0 (disabled) or an even integer from 2 to 64")
    return value


def replay_policy(parity):
    validate_rs_parity(parity)
    return {"enabled": parity != 0, "metric": "recorded-error-replay",
            "parityBytesPerBlock": parity, "maximumBlockBytes": 255,
            "crc": "CRC-32/ISO-HDLC", "crcBytesPerPacket": 4,
            "packing": "masked-data-row-major/high-nibble-first/drop-last-unpaired-cell",
            "blocks": "balanced-shortened/round-robin-interleaved",
            "erasures": False,
            "note": "Diagnostic replay of observed byte XOR errors on encoded packets, including parity and CRC positions. This assumes the error pattern transfers to changed symbols; it is not measured packet reception or an authentication check. Raw exactness and checkpoint selection are unchanged."}


def replay_summary(counter):
    if not counter.get("rsReplayCases", 0):
        return None
    result = {name: counter.get(field, 0) for field, name in REPLAY_COUNTERS.items()}
    eligible = result["eligibleCases"]
    result["ineligibleCases"] = result["cases"] - eligible
    for rate, count in (("rsDecodedRate", "rsDecodedCases"),
                        ("crcAcceptedRate", "crcAcceptedCases"), ("exactRate", "exactCases")):
        result[rate] = result[count]/eligible if eligible else None
    return result


def pack_data_cells(values, mask):
    values, mask = np.asarray(values), np.asarray(mask)
    if (values.shape != mask.shape or values.ndim != 2 or mask.dtype != np.bool_
            or not np.issubdtype(values.dtype, np.integer)):
        raise ValueError("Replay requires integer symbol matrices and their boolean data mask")
    data = values[mask]
    if np.any(data < 0) or np.any(data > 15):
        raise ValueError("Replay requires a sixteen-color logical label space")
    data = data[:len(data)//2*2].astype(np.uint8, copy=False)
    return ((data[::2] << 4) | data[1::2]).tobytes()


class ReplayBatch:
    """A bounded-memory replay job; one response is consumed at a time."""

    def __init__(self, parity=32, *, node="node", work_dir=None):
        self.parity = validate_rs_parity(parity)
        if parity == 0:
            raise ValueError("Do not create a replay batch when rs-parity is zero")
        self.node = node
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="rs-replay-", dir=work_dir)
        self.root = Path(self.temp.name)
        self.requests = (self.root/"requests.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.requests.close()
        self.temp.cleanup()

    def append(self, expected, predicted, mask):
        if self.requests.closed:
            raise RuntimeError("Replay requests are already closed")
        row = {"id": str(self.count),
               "expected": base64.b64encode(pack_data_cells(expected, mask)).decode("ascii"),
               "predicted": base64.b64encode(pack_data_cells(predicted, mask)).decode("ascii")}
        self.requests.write(json.dumps(row, separators=(",", ":"))+"\n")
        self.count += 1
        return self.count-1

    def results(self):
        self.requests.close()
        with (self.root/"requests.jsonl").open("r", encoding="utf-8") as input_, \
                (self.root/"responses.jsonl").open("w", encoding="utf-8") as output, \
                (self.root/"stderr.txt").open("w+", encoding="utf-8") as errors:
            try:
                process = subprocess.run([self.node, str(ROOT/"scripts/corrector-rs-replay.mjs"),
                                          "--parity", str(self.parity)],
                                         stdin=input_, stdout=output, stderr=errors, text=True)
            except OSError as error:
                raise RuntimeError(f"Cannot start canonical RS replay; check --node or use --rs-parity 0: {error}") from error
            if process.returncode:
                errors.seek(0)
                raise RuntimeError(f"Canonical RS replay failed: {errors.read()[-2000:]}")
        with (self.root/"responses.jsonl").open(encoding="utf-8") as stream:
            seen = 0
            for line in stream:
                row = json.loads(line)
                if row.get("id") != str(seen) or seen >= self.count:
                    raise RuntimeError("Canonical RS replay returned a different case order")
                flags = ("eligible", "rawCRCValid", "rsDecoded", "crcAccepted", "exact", "wrongAccepted")
                counts = ("payloadBytes", "wireBytes", "parityBytes", "blockCount")
                if any(type(row.get(name)) is not int or row[name] not in (0, 1) for name in flags):
                    raise RuntimeError("Canonical RS replay returned invalid flags")
                if any(type(row.get(name)) is not int or row[name] < 0 for name in counts):
                    raise RuntimeError("Canonical RS replay returned invalid byte counts")
                if not (row["exact"] + row["wrongAccepted"] == row["crcAccepted"]
                        <= row["rsDecoded"] <= row["eligible"]):
                    raise RuntimeError("Canonical RS replay returned inconsistent acceptance")
                seen += 1
                yield row
            if seen != self.count:
                raise RuntimeError("Canonical RS replay omitted cases")


def add_replay(counter, row):
    mapping = {"rsReplayCases": 1, "rsEligibleCases": row["eligible"],
               "rsRawCRCValidCases": row["rawCRCValid"], "rsDecodedCases": row["rsDecoded"],
               "rsCRCAcceptedCases": row["crcAccepted"], "rsExactCases": row["exact"],
               "rsWrongAcceptedCases": row["wrongAccepted"],
               "rsPayloadBytes": row["payloadBytes"] if row["eligible"] else 0,
               "rsWireBytes": row["wireBytes"] if row["eligible"] else 0,
               "rsParityBytes": row["parityBytes"] if row["eligible"] else 0,
               "rsBlocks": row["blockCount"] if row["eligible"] else 0}
    for key, value in mapping.items():
        counter[key] = counter.get(key, 0) + value
