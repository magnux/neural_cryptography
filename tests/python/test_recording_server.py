"""Recording persistence and HTTP boundaries; no camera/model dependencies."""
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("recording_server", ROOT / "apps/color-optical-bench/serve.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def profile(size=64):
    return {"size": size, "colors": 16, "phases": 4, "gridBlock": 8,
            "permute": True, "pixelsPerCell": 2, "holdMs": 4000, "parity": 0}


def info(size=64, phase=0):
    return {**profile(size), "phase": phase, "bits": 4,
            "collectionCode": "1234abcd", "id": "1234abcd"}


def sender(size=64):
    return {"format": "neural-qr-sender", "version": 1,
            "created": "2026-11-04T12:34:56.789Z", "sessionId": "sender-abc",
            "datasetSplit": "train", "collectionCode": "1234abcd",
            "payloadId": "a" * 64, "payloadBase64": "AQID",
            "profile": profile(size),
            "environment": {"deviceId": "device-one", "browser": {"name": "test"}},
            "frames": [{"info": info(size, phase), "key": f"1234abcd/{size}/16/8/{phase}"} for phase in range(4)]}


def receiver(size=64):
    return {"format": "neural-qr-captures", "version": 1,
            "created": "2026-11-04T12:34:56.789Z", "sessionId": "receiver-abc",
            "datasetSplit": "train", "collectionCode": "1234abcd", "profile": profile(size),
            "environment": {"deviceId": "device-two"},
            "camera": {"label": "Front camera", "deviceId": "origin-scoped-camera"},
            "records": [{"ok": True, "info": info(size, phase), "pixelsPerCell": 2,
                         "key": f"1234abcd/{size}/16/8/{phase}",
                         "inputBase64": "AA==", "samplesBase64": "AA==",
                         "stability": {"guardMs": 500, "durationMs": 600, "observations": 3}} for phase in range(4)],
            "diagnostics": []}

class QuietHandler(MODULE.BenchHandler):
    def log_message(self, *args):
        pass


class RecordingServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.dist / "index.html").write_text("collector test", encoding="utf-8")
        self.store = MODULE.RecordingStore(self.root / "recordings", self.dist, max_bytes=4096)

    @contextmanager
    def server(self, enabled=True):
        application = partial(QuietHandler, directory=str(self.dist), recordings=self.store if enabled else None)
        server = ThreadingHTTPServer(("127.0.0.1", 0), application)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.port = server.server_address[1]
        try:
            yield
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def request(self, method, path, value=None, headers=None):
        body = json.dumps(value).encode() if isinstance(value, dict) else value
        supplied = {"Origin": f"http://127.0.0.1:{self.port}", "Sec-Fetch-Site": "same-origin"}
        if method == "POST":
            supplied.update({"Content-Type": "application/json", "X-Upload-Token": self.store.token})
        supplied.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            raw = response.read()
            result = json.loads(raw) if "application/json" in response.getheader("Content-Type", "") else raw
            return response.status, result
        finally:
            connection.close()

    def test_config_requires_opt_in_and_no_public_recording_reads(self):
        with self.server(enabled=False):
            status, config = self.request("GET", "/api/recordings/config")
            self.assertEqual(status, 200)
            self.assertFalse(config["enabled"])
            self.assertNotIn("uploadToken", config)
            self.assertEqual(self.request("POST", "/api/recordings?split=train", sender())[0], 404)
        with self.server():
            status, config = self.request("GET", "/api/recordings/config")
            self.assertEqual(status, 200)
            self.assertEqual(config, {"enabled": True, "maxUploadBytes": 4096, "uploadToken": self.store.token})
            self.assertEqual(self.request("GET", "/api/recordings")[0], 404)
            (self.dist / "leak").symlink_to(self.store.root, target_is_directory=True)
            self.assertEqual(self.request("GET", "/leak/")[0], 404)

    def test_sender_and_receiver_upload_group_by_split_size_and_keep_metadata(self):
        with self.server():
            for make, role in ((sender, "sender"), (receiver, "receiver")):
                recording = make(256)
                recording["datasetSplit"] = "validation"
                status, result = self.request("POST", "/api/recordings?split=validation", recording)
                self.assertEqual(status, 201, result)
                self.assertTrue(result["storedPath"].startswith(f"validation/g256/{role}/1234abcd_neural-qr-"))
                saved = self.store.root / result["storedPath"]
                self.assertEqual(json.loads(saved.read_text()), recording)
                self.assertFalse(result["duplicate"])
                self.assertEqual(result["bytes"], saved.stat().st_size)
                self.assertEqual(len(result["sha256"]), 64)

    def test_duplicate_is_idempotent_and_different_content_never_overwrites(self):
        with self.server():
            original = receiver()
            status, first = self.request("POST", "/api/recordings?split=train", original)
            self.assertEqual(status, 201)
            status, repeat = self.request("POST", "/api/recordings?split=train", original)
            self.assertEqual(status, 200)
            self.assertTrue(repeat["duplicate"])
            self.assertEqual(first["sha256"], repeat["sha256"])
            changed = deepcopy(original)
            changed["camera"]["label"] = "different camera"
            self.assertEqual(self.request("POST", "/api/recordings?split=train", changed)[0], 409)
            self.assertEqual(json.loads((self.store.root / first["storedPath"]).read_text()), original)
            self.assertFalse(list(self.store.root.rglob(".upload-*")))

    def test_unsafe_origin_token_content_type_or_split_reject(self):
        with self.server():
            for headers, expected in (({"Origin": "https://other.example"}, 403),
                                      ({"Sec-Fetch-Site": "cross-site"}, 403),
                                      ({"X-Upload-Token": "wrong"}, 403),
                                      ({"Content-Type": "text/plain"}, 415)):
                self.assertEqual(self.request("POST", "/api/recordings?split=train", sender(), headers)[0], expected)
            for path in ("/api/recordings?split=../outside", "/api/recordings?split=train&split=test", "/api/recordings"):
                self.assertEqual(self.request("POST", path, sender())[0], 422)
            self.assertEqual(self.request("POST", "/api/recordings?split=test", sender())[0], 422)
            self.assertEqual(self.request("GET", "/api/recordings/config", headers={"Origin": "https://other.example"})[0], 403)
        self.assertFalse(list(self.store.root.rglob("*.json")))

    def test_oversize_invalid_json_and_concurrent_upload_limit(self):
        with self.server():
            self.assertEqual(self.request("POST", "/api/recordings?split=train", b" " * 4097)[0], 413)
            self.assertEqual(self.request("POST", "/api/recordings?split=train", b'{"x":NaN}')[0], 400)
            self.assertEqual(self.request("POST", "/api/recordings?split=train", b"\xff")[0], 400)
            self.store.slots.acquire()
            self.store.slots.acquire()
            try:
                self.assertEqual(self.request("POST", "/api/recordings?split=train", sender())[0], 429)
            finally:
                self.store.slots.release()
                self.store.slots.release()

    def test_mixed_grid_invalid_identity_and_unknown_diagnostics(self):
        with self.server():
            mixed = receiver()
            mixed["records"][1]["info"]["size"] = 128
            self.assertEqual(self.request("POST", "/api/recordings?split=train", mixed)[0], 422)
            unsafe = receiver()
            unsafe["sessionId"] = "../../outside"
            self.assertEqual(self.request("POST", "/api/recordings?split=train", unsafe)[0], 422)
            invalid = sender()
            invalid["frames"][0]["info"]["collectionCode"] = "bad"
            self.assertEqual(self.request("POST", "/api/recordings?split=train", invalid)[0], 422)
            diagnostic = receiver()
            diagnostic["records"] = []
            diagnostic["diagnostics"] = [{"stage": "markers", "error": "not found"}]
            status, result = self.request("POST", "/api/recordings?split=train", diagnostic)
            self.assertEqual(status, 201, result)
            self.assertTrue(result["storedPath"].startswith("train/unknown/receiver/1234abcd_"))

    def test_current_profile_code_and_phase_membership_are_required(self):
        with self.server():
            mutations = [
                lambda b: b.update(format="optical-decoder-sender"),
                lambda b: b.update(collectionCode="ABCDEF12"),
                lambda b: b["profile"].update(colors=4),
                lambda b: b["profile"].update(pixelsPerCell=4),
                lambda b: b["profile"].update(phases=4.0),
                lambda b: b["frames"][0]["info"].update(phase=True),
                lambda b: b["frames"][0].update(key="wrong"),
                lambda b: b["frames"].__setitem__(1, deepcopy(b["frames"][0])),
            ]
            for change in mutations:
                recording = sender()
                change(recording)
                with self.subTest(recording=recording):
                    self.assertEqual(self.request("POST", "/api/recordings?split=train", recording)[0], 422)
            malformed = receiver()
            malformed["records"][0]["pixelsPerCell"] = 4
            self.assertEqual(self.request("POST", "/api/recordings?split=train", malformed)[0], 422)
        self.assertFalse(list(self.store.root.rglob("*.json")))

    def test_storage_must_be_private_and_symlink_cannot_redirect_writes(self):
        for location in (self.dist, self.dist / "uploads", self.dist.parent):
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "outside"):
                MODULE.RecordingStore(location, self.dist)
        outside = self.root / "outside"
        outside.mkdir()
        (self.store.root / "train").symlink_to(outside, target_is_directory=True)
        with self.server():
            self.assertEqual(self.request("POST", "/api/recordings?split=train", sender())[0], 409)
        self.assertEqual(list(outside.iterdir()), [])


    def test_collection_stages_sender_and_exposes_only_public_assignment(self):
        with self.server():
            self.assertEqual(self.request("GET", "/api/collection-session")[1], {"enabled": True, "session": None})
            original = sender()
            status, started = self.request("POST", "/api/collection-session?split=train", original)
            self.assertEqual(status, 201, started)
            self.assertTrue(started["storedPath"].endswith(".json.tmp"))
            staged = self.store.root / started["storedPath"]
            self.assertEqual(json.loads(staged.read_text()), original)
            self.assertFalse(staged.with_suffix("").exists())
            status, observed = self.request("GET", "/api/collection-session")
            descriptor = observed["session"]
            self.assertEqual(set(descriptor), {"format", "version", "id", "collectionCode", "senderSessionId", "profile", "datasetSplit", "created"})
            self.assertEqual(descriptor, started["session"])
            self.assertEqual(json.loads(self.store.session_path.read_text()), descriptor)
            self.assertNotIn("payload", json.dumps(descriptor))
            self.assertNotIn("environment", descriptor)
            status, repeat = self.request("POST", "/api/collection-session?split=train", original)
            self.assertEqual(status, 200)
            self.assertTrue(repeat["duplicate"])
            self.assertEqual(repeat["session"], descriptor)
            other = sender(128)
            self.assertEqual(self.request("POST", "/api/collection-session?split=train", other)[0], 409)
            changed = deepcopy(original)
            changed["environment"]["deviceId"] = "changed"
            self.assertEqual(self.request("POST", "/api/collection-session?split=train", changed)[0], 409)
            restarted = MODULE.RecordingStore(self.store.root, self.dist)
            self.assertEqual(restarted.current_session(), descriptor)

    def test_matching_receiver_finalizes_sender_and_retries_are_idempotent(self):
        with self.server():
            original = sender()
            _, started = self.request("POST", "/api/collection-session?split=train", original)
            captured = receiver()
            captured["collectionSessionId"] = started["session"]["id"]
            status, saved = self.request("POST", "/api/recordings?split=train", captured)
            self.assertEqual(status, 201, saved)
            self.assertTrue(saved["sessionCompleted"])
            self.assertFalse(self.store.session_path.exists())
            staged = self.store.root / started["storedPath"]
            self.assertFalse(staged.exists())
            self.assertEqual(json.loads(staged.with_suffix("").read_text()), original)
            self.assertEqual(self.request("GET", "/api/collection-session")[1]["session"], None)
            status, retry = self.request("POST", "/api/recordings?split=train", captured)
            self.assertEqual(status, 200)
            self.assertTrue(retry["duplicate"])
            self.assertFalse(retry["sessionCompleted"])
            self.assertTrue(retry["sessionInactive"])
            status, late_sender = self.request("POST", "/api/collection-session?split=train", original)
            self.assertEqual(status, 200)
            self.assertTrue(late_sender["sessionCompleted"])
            self.assertIsNone(late_sender["session"])
            captured["created"] = "2026-11-05T12:34:56.789Z"
            self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 409)

    def test_failed_or_unrelated_receiver_never_completes_session(self):
        with self.server():
            _, started = self.request("POST", "/api/collection-session?split=train", sender())
            identifier = started["session"]["id"]
            for captured in (receiver(128), receiver()):
                captured["collectionSessionId"] = identifier
                if captured["profile"]["size"] == 64:
                    captured["records"][0].pop("stability")
                self.assertIn(self.request("POST", "/api/recordings?split=train", captured)[0], (409, 422))
            captured = receiver()
            captured["collectionSessionId"] = "wrong-id"
            self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 409)
            captured["collectionSessionId"] = identifier
            captured["records"] = []
            captured["diagnostics"] = [{"stage": "phase"}]
            self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 422)
            self.assertTrue((self.store.root / started["storedPath"]).exists())
            self.assertEqual(self.store.current_session()["id"], identifier)
            # Existing smaller recordings have no session field and remain valid.
            captured = receiver()
            captured["records"] = captured["records"][:1]
            self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 201)
            self.assertEqual(self.store.current_session()["id"], identifier)

    def test_abandon_preserves_staged_data_and_stale_completion_cannot_clear_new_session(self):
        with self.server():
            _, started = self.request("POST", "/api/collection-session?split=train", sender())
            self.assertEqual(self.request("POST", "/api/collection-session/abandon", {"id": "wrong"})[0], 409)
            status, result = self.request("POST", "/api/collection-session/abandon", {"id": started["session"]["id"]})
            self.assertEqual(status, 200)
            self.assertTrue(result["abandoned"])
            self.assertTrue((self.store.root / started["storedPath"]).exists())
            self.assertFalse(self.store.session_path.exists())
            _, replacement = self.request("POST", "/api/collection-session?split=train", sender(128))
            captured = receiver()
            captured["collectionSessionId"] = started["session"]["id"]
            self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 409)
            self.assertEqual(self.store.current_session()["id"], replacement["session"]["id"])

    def test_finalization_failure_keeps_receiver_and_retry_completes_without_rewriting_it(self):
        with self.server():
            _, started = self.request("POST", "/api/collection-session?split=train", sender())
            captured = receiver()
            captured["collectionSessionId"] = started["session"]["id"]
            with mock.patch.object(self.store, "_finish_session", side_effect=OSError("disk unavailable")):
                self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 507)
            self.assertTrue(self.store.session_path.exists())
            self.assertTrue((self.store.root / started["storedPath"]).exists())
            saved = self.store.root / MODULE.recording_destination(captured, "train")
            original_inode = saved.stat().st_ino
            status, retry = self.request("POST", "/api/recordings?split=train", captured)
            self.assertEqual(status, 200, retry)
            self.assertTrue(retry["sessionCompleted"])
            self.assertEqual(saved.stat().st_ino, original_inode)

    def test_partial_finalize_then_abandon_reports_saved_only_on_duplicate(self):
        with self.server():
            _, started = self.request("POST", "/api/collection-session?split=train", sender())
            captured = receiver()
            captured["collectionSessionId"] = started["session"]["id"]
            with mock.patch.object(self.store, "_finish_session", side_effect=OSError("disk unavailable")):
                self.assertEqual(self.request("POST", "/api/recordings?split=train", captured)[0], 507)
            self.assertEqual(self.request("POST", "/api/collection-session/abandon", {"id": started["session"]["id"]})[0], 200)
            _, replacement = self.request("POST", "/api/collection-session?split=train", sender(128))
            status, retry = self.request("POST", "/api/recordings?split=train", captured)
            self.assertEqual(status, 200, retry)
            self.assertTrue(retry["duplicate"])
            self.assertFalse(retry["sessionCompleted"])
            self.assertTrue(retry["sessionInactive"])
            self.assertTrue((self.store.root / started["storedPath"]).exists())
            self.assertEqual(self.store.current_session()["id"], replacement["session"]["id"])

    def test_session_endpoints_reject_unsafe_requests_and_keep_disabled_mode(self):
        with self.server(enabled=False):
            self.assertEqual(self.request("GET", "/api/collection-session")[1], {"enabled": False, "session": None})
            self.assertEqual(self.request("POST", "/api/collection-session?split=train", sender())[0], 404)
        with self.server():
            self.assertEqual(self.request("GET", "/api/collection-session", headers={"Origin": "https://other.example"})[0], 403)
            self.assertEqual(self.request("POST", "/api/collection-session?split=train", sender(), {"X-Upload-Token": "bad"})[0], 403)
            self.assertEqual(self.request("POST", "/api/collection-session?split=train", receiver())[0], 422)
            self.assertEqual(self.request("POST", "/api/collection-session/abandon", {"id": "bad", "extra": 1})[0], 422)
        self.assertEqual(MODULE.MAX_UPLOAD_BYTES, 800 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
