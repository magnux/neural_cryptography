#!/usr/bin/env python3
"""Create explicit transmission-direction groups from current capture exports.

The flags assign direction; filenames, camera labels and UUIDs never determine it.
Only receiver session identities are copied. Recordings and joined datasets are
read without modification, and all real dataset sessions must be covered.
"""

import argparse
import json
from pathlib import Path
import re

from data import Dataset
from sampling import validate_session_groups


MAX_CAPTURE_JSON = (160 + 4) * 1024 * 1024
SESSION_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$")


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def capture_session(path):
    """Read one bounded current capture export, returning its receiver identity."""
    path = Path(path)
    if path.stat().st_size > MAX_CAPTURE_JSON:
        raise ValueError(f"Capture exceeds the current 164 MiB JSON limit: {path}")
    with path.open(encoding="utf8") as file:
        bundle = json.load(file, object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    if (not isinstance(bundle, dict) or bundle.get("format") != "optical-decoder-captures"
            or type(bundle.get("version")) is not int or bundle["version"] != 1
            or not isinstance(bundle.get("records"), list) or len(bundle["records"]) > 128):
        raise ValueError(f"Unsupported current receiver capture bundle: {path}")
    session = bundle.get("sessionId")
    if not isinstance(session, str) or not SESSION_ID.fullmatch(session):
        raise ValueError(f"Invalid receiver session ID: {path}")
    return session


def create_session_groups(datasets, out, *, pc_to_phone=(), phone_to_pc=()):
    """Validate explicit assignments and write a new sidecar, never overwriting."""
    out = Path(out)
    if out.exists():
        raise ValueError("Session-group output already exists; choose a new output path")
    sessions = {}
    for direction, paths in (("pc-to-phone", pc_to_phone), ("phone-to-pc", phone_to_pc)):
        for path in paths:
            session = capture_session(path)
            if session in sessions and sessions[session] != direction:
                raise ValueError(f"Receiver session assigned to conflicting directions: {session}")
            sessions[session] = direction
    if not sessions:
        raise ValueError("Provide capture files for at least one transmission direction")
    dataset = Dataset(datasets)
    validate_session_groups(dataset.samples, sessions)
    value = {"format": "optical-decoder-session-groups", "version": 1,
             "sessions": dict(sorted(sessions.items()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also refuses an output created after the first check.
    with out.open("x", encoding="utf8") as file:
        file.write(json.dumps(value, indent=2)+"\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True,
                        help="joined train/validation datasets whose real sessions must be mapped")
    parser.add_argument("--pc-to-phone", type=Path, nargs="+", default=[],
                        help="receiver capture exports recorded on the phone")
    parser.add_argument("--phone-to-pc", type=Path, nargs="+", default=[],
                        help="receiver capture exports recorded on the PC")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = create_session_groups(args.datasets, args.out,
                                       pc_to_phone=args.pc_to_phone, phone_to_pc=args.phone_to_pc)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Session grouping failed: {error}\n")
    print(json.dumps({"out": str(args.out.resolve()), "sessions": len(result["sessions"]),
                      "groups": {group: sum(value == group for value in result["sessions"].values())
                                 for group in sorted(set(result["sessions"].values()))}}, indent=2))


if __name__ == "__main__":
    main()
