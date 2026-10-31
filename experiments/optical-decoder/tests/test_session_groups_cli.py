"""Explicit direction sidecars leave capture files and dataset formats intact."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))

from sampling import load_session_groups
from session_groups import capture_session, create_session_groups


class SessionGroupTests(unittest.TestCase):
    def setUp(self):
        root = ROOT/"validation/tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="session-groups-", dir=root)
        self.directory = Path(self.temporary.name)
        self.phone = self.capture("misleading-pc-filename.json", "receiver-phone")
        self.pc = self.capture("misleading-phone-filename.json", "receiver-pc")
        self.dataset = self.directory/"dataset"
        self.dataset.mkdir()
        samples = []
        for index, session in enumerate(("receiver-phone", "receiver-pc")):
            samples.append({"id": f"sample-{index}", "sessionId": session,
                            "payloadId": f"{index+1:064x}", "source": "real", "split": "train",
                            "size": 64, "colors": 16, "info": {"size": 64, "colors": 16},
                            **{name: f"{name}.bin" for name in ("input", "labels", "mask", "baseline", "expected")},
                            "hashes": {name: "a"*64 for name in ("input", "labels", "mask", "baseline", "expected")}})
        self.manifest = self.dataset/"manifest.json"
        self.manifest.write_text(json.dumps({"format": "optical-decoder-dataset", "version": 1,
                                             "samples": samples}))
        self.out = self.directory/"groups.json"

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, filename, session):
        path = self.directory/filename
        path.write_text(json.dumps({"format": "optical-decoder-captures", "version": 1,
                                    "sessionId": session, "records": []}))
        return path

    def test_explicit_assignments_deduplicate_and_preserve_inputs(self):
        inputs = {path: path.read_bytes() for path in (self.phone, self.pc, self.manifest)}
        extra = self.capture("unused.json", "receiver-extra")
        result = create_session_groups([self.dataset], self.out,
                                       pc_to_phone=[self.phone, self.phone, extra], phone_to_pc=[self.pc])
        self.assertEqual(result["sessions"], {"receiver-phone": "pc-to-phone",
                                             "receiver-pc": "phone-to-pc",
                                             "receiver-extra": "pc-to-phone"})
        self.assertEqual(load_session_groups(self.out), result["sessions"])
        for path, before in inputs.items():
            self.assertEqual(path.read_bytes(), before)

    def test_cli_creates_loadable_mapping(self):
        run = subprocess.run([sys.executable, str(EXPERIMENT/"session_groups.py"),
                              "--datasets", str(self.dataset), "--pc-to-phone", str(self.phone),
                              "--phone-to-pc", str(self.pc), "--out", str(self.out)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["groups"], {"pc-to-phone": 1, "phone-to-pc": 1})
        self.assertEqual(len(load_session_groups(self.out)), 2)

    def test_conflicts_and_missing_sessions_fail_before_writing(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            create_session_groups([self.dataset], self.out,
                                  pc_to_phone=[self.phone], phone_to_pc=[self.phone, self.pc])
        self.assertFalse(self.out.exists())
        with self.assertRaises(ValueError):
            create_session_groups([self.dataset], self.out, pc_to_phone=[self.phone])
        self.assertFalse(self.out.exists())

    def test_existing_output_is_never_overwritten(self):
        self.out.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_session_groups([self.dataset], self.out,
                                  pc_to_phone=[self.phone], phone_to_pc=[self.pc])
        self.assertEqual(self.out.read_text(), "keep me")

    def test_capture_format_identity_and_duplicate_fields_are_checked(self):
        original = json.loads(self.phone.read_text())
        cases = [{**original, "version": True}, {**original, "version": 2},
                 {**original, "format": "other"}, {**original, "sessionId": 2},
                 {**original, "sessionId": "bad/session"}, {**original, "records": {}},
                 {**original, "records": [None]*129}]
        for value in cases:
            with self.subTest(value=value):
                self.phone.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    capture_session(self.phone)
        self.phone.write_text('{"format":"optical-decoder-captures","version":1,"sessionId":"a","sessionId":"b","records":[]}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            capture_session(self.phone)


if __name__ == "__main__":
    unittest.main()
