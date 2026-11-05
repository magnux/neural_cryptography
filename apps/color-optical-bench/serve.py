#!/usr/bin/env python3
"""Serve the optical bench and optional recording uploads. Standard library only.

Local webcam: python serve.py
Phone/LAN:   python serve.py --host 0.0.0.0 --https --cert certs/dev.pem --key certs/dev-key.pem
The phone must trust the certificate. Add --recordings-dir local-data/recordings
to save explicitly uploaded collector JSONs outside the served application.
This server performs no optical decoding, model inference or authentication.
"""
import argparse
from datetime import datetime
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import ssl
import stat
import tempfile
import threading
import uuid
from urllib.parse import parse_qs, urlsplit


MAX_UPLOAD_BYTES = 800 * 1024 * 1024
MAX_SESSION_BYTES = 1024 * 1024
SIZES = (8, 16, 32, 64, 128, 256)
SPLITS = ("train", "validation", "test")


class RecordingError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def recording_destination(bundle, split):
    """Validate the storage envelope; the offline join validates tensor truth.

    Paths are derived exclusively from validated identity fields, never a
    client filename. Optional environment/camera metadata stays in the JSON.
    """
    if split not in SPLITS:
        raise RecordingError("Choose train, validation or test.")
    if not isinstance(bundle, dict) or type(bundle.get("version")) is not int or bundle["version"] != 1:
        raise RecordingError("Expected a current recording JSON.")
    if bundle.get("datasetSplit", split) != split:
        raise RecordingError("Recording datasetSplit does not match the selected upload split.")
    session = bundle.get("sessionId")
    if not isinstance(session, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", session):
        raise RecordingError("Invalid physical session ID.")
    created = bundle.get("created")
    if not isinstance(created, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", created):
        raise RecordingError("Invalid recording creation timestamp.")
    try:
        datetime.fromisoformat(created[:-1] + "+00:00")
    except ValueError as error:
        raise RecordingError("Invalid recording creation timestamp.") from error
    code = bundle.get("collectionCode")
    if not isinstance(code, str) or not re.fullmatch(r"[a-f0-9]{8}", code):
        raise RecordingError("Invalid public collection code.")
    profile = bundle.get("profile")
    required = {"colors": 16, "phases": 4, "gridBlock": 8, "pixelsPerCell": 2,
                "holdMs": 4000, "parity": 0}
    if (not isinstance(profile, dict) or type(profile.get("size")) is not int
            or profile["size"] not in SIZES or profile.get("permute") is not True
            or any(type(profile.get(key)) is not int or profile[key] != value
                   for key, value in required.items())):
        raise RecordingError("Unsupported Neural-QR collection profile.")
    kind = bundle.get("format")
    if kind == "neural-qr-sender":
        role, rows = "sender", bundle.get("frames")
        if not isinstance(rows, list) or len(rows) != 4:
            raise RecordingError("Expected four sender phases.")
        if not isinstance(bundle.get("payloadId"), str) or not re.fullmatch(r"[a-f0-9]{64}", bundle["payloadId"]):
            raise RecordingError("Invalid sender payload identity.")
        if not isinstance(bundle.get("payloadBase64"), str):
            raise RecordingError("Missing sender payload bytes.")
    elif kind == "neural-qr-captures":
        role, rows = "receiver", bundle.get("records")
        diagnostics = bundle.get("diagnostics", [])
        if not isinstance(rows, list) or len(rows) > 128 or not isinstance(diagnostics, list) or len(diagnostics) > 100:
            raise RecordingError("Invalid capture or diagnostic count.")
        if not rows and not diagnostics:
            raise RecordingError("Recording contains no captures or diagnostics.")
    else:
        raise RecordingError("Only current Neural-QR sender and receiver recording JSONs can be uploaded.")
    seen_phases = set()
    for row in rows:
        info = row.get("info") if isinstance(row, dict) else None
        if not isinstance(info, dict):
            raise RecordingError("Recording contains invalid matrix metadata.")
        phase = info.get("phase")
        if (type(phase) is not int or not 0 <= phase < 4 or info.get("collectionCode") != code
                or info.get("id") != code or type(info.get("bits")) is not int or info["bits"] != 4
                or any(type(info.get(key)) is not type(value) or info[key] != value
                       for key, value in profile.items())):
            raise RecordingError("Recording metadata disagrees with its collection code or profile.")
        if row.get("key") != f"{code}/{profile['size']}/16/8/{phase}":
            raise RecordingError("Recording key disagrees with its phase metadata.")
        if role == "sender":
            if phase in seen_phases:
                raise RecordingError("Sender phase list contains duplicates.")
            seen_phases.add(phase)
        elif (row.get("ok") is not True or type(row.get("pixelsPerCell")) is not int
              or row["pixelsPerCell"] != 2
              or not all(isinstance(row.get(key), str) for key in ("inputBase64", "samplesBase64"))):
            raise RecordingError("Missing receiver evidence.")
    grid = f"g{profile['size']}" if rows else "unknown"
    filename = (f"{code}_neural-qr-sender-{session}.json" if role == "sender" else
                f"{code}_neural-qr-captures-{session}-{created.replace(':', '-').replace('.', '-')}.json")
    return Path(split, grid, role, filename)


class RecordingStore:
    def __init__(self, root, served_root, max_bytes=MAX_UPLOAD_BYTES):
        self.root = Path(root).expanduser().resolve()
        served_root = Path(served_root).resolve()
        if self.root == served_root or served_root in self.root.parents or self.root in served_root.parents:
            raise ValueError("Recording storage must be outside and separate from the served dist directory, not its parent.")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= MAX_UPLOAD_BYTES:
            raise ValueError("Invalid recording upload limit.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes = max_bytes
        self.token = secrets.token_urlsafe(32)
        self.slots = threading.BoundedSemaphore(2)
        self.session_lock = threading.RLock()
        self.session_path = self.root / "collection_session.json"

    def _save_bytes(self, relative, body):
        parent = self.root
        for part in relative.parts[:-1]:
            parent = parent / part
            parent.mkdir(exist_ok=True, mode=0o700)
            if parent.is_symlink() or not parent.is_dir() or self.root not in parent.resolve().parents:
                raise RecordingError("Recording storage contains an unsafe directory.", 409)
        target = parent / relative.name
        digest = hashlib.sha256(body).hexdigest()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".upload-", dir=parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            try:
                # Atomic no-replace publication: a retry can never truncate a recording.
                os.link(temporary, target)
                duplicate = False
            except FileExistsError:
                if target.is_symlink():
                    raise RecordingError("Recording destination is not a regular file.", 409)
                descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as existing:
                    details = os.fstat(existing.fileno())
                    if not stat.S_ISREG(details.st_mode) or details.st_size != len(body):
                        raise RecordingError("A different recording already uses this filename; nothing was overwritten.", 409)
                    current = hashlib.sha256()
                    while block := existing.read(1024 * 1024):
                        current.update(block)
                if current.hexdigest() != digest:
                    raise RecordingError("A different recording already uses this filename; nothing was overwritten.", 409)
                duplicate = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"ok": True, "storedPath": relative.as_posix(), "bytes": len(body),
                "sha256": digest, "duplicate": duplicate}


    @staticmethod
    def _sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read_session(self):
        if not self.session_path.exists() and not self.session_path.is_symlink():
            return None
        if self.session_path.is_symlink() or not self.session_path.is_file() or self.session_path.stat().st_size > 65536:
            raise RecordingError("Invalid collection session metadata; inspect server storage.", 409)
        try:
            value = json.loads(self.session_path.read_text(encoding="utf-8"))
            self._sender_relative(value)
        except (ValueError, TypeError, KeyError) as error:
            raise RecordingError("Invalid collection session metadata; inspect server storage.", 409) from error
        return value

    @staticmethod
    def _sender_relative(session):
        # The file contains only this public descriptor. No target data is served.
        fields = {"format", "version", "id", "collectionCode", "senderSessionId", "profile", "datasetSplit", "created"}
        if (not isinstance(session, dict) or set(session) != fields
                or session["format"] != "neural-qr-collection-session" or session["version"] != 1
                or not isinstance(session["id"], str)
                or not re.fullmatch(r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}", session["id"])):
            raise ValueError("Invalid session descriptor")
        profile, code = session["profile"], session["collectionCode"]
        # Reuse the current recording envelope validator to check every path/profile field.
        frames = [{"key": f"{code}/{profile['size']}/16/8/{phase}",
                   "info": {**profile, "collectionCode": code, "id": code, "phase": phase, "bits": 4}}
                  for phase in range(4)]
        bundle = {"format": "neural-qr-sender", "version": 1, "sessionId": session["senderSessionId"],
                  "created": session["created"], "collectionCode": code, "profile": profile,
                  "datasetSplit": session["datasetSplit"], "payloadId": "0" * 64,
                  "payloadBase64": "", "frames": frames}
        return recording_destination(bundle, session["datasetSplit"])

    def _write_session(self, descriptor):
        if self.session_path.is_symlink():
            raise RecordingError("Collection session metadata must not be a symlink.", 409)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".collection-", dir=self.root, delete=False) as output:
                temporary = Path(output.name)
                output.write(json.dumps(descriptor, separators=(",", ":")).encode("utf-8"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.session_path)
            self._sync_directory(self.root)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _existing_matches(self, relative, body):
        path = self.root / relative
        if not path.exists() and not path.is_symlink():
            return False
        if path.is_symlink() or not path.is_file() or self.root not in path.resolve().parents or any(parent.is_symlink() for parent in path.parents if parent != self.root and self.root in parent.parents):
            raise RecordingError("Recording destination is not a regular file.", 409)
        if path.stat().st_size != len(body):
            raise RecordingError("A different recording already uses this filename; nothing was overwritten.", 409)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != hashlib.sha256(body).hexdigest():
            raise RecordingError("A different recording already uses this filename; nothing was overwritten.", 409)
        return True

    def current_session(self):
        with self.session_lock:
            return self._read_session()

    def begin_session(self, bundle, split, body):
        relative = recording_destination(bundle, split)
        if bundle["format"] != "neural-qr-sender":
            raise RecordingError("Start collection with a sender manifest.")
        if len(body) > MAX_SESSION_BYTES:
            raise RecordingError("Sender manifest exceeds the collection session limit.", 413)
        staged = relative.with_name(relative.name + ".tmp")
        with self.session_lock:
            active = self._read_session()
            if active is not None:
                if self._sender_relative(active) != relative:
                    raise RecordingError("Another matrix is still being collected. Finish or abandon that collection first.", 409)
                source = staged if (self.root / staged).exists() else relative
                if not self._existing_matches(source, body):
                    raise RecordingError("Active sender file is missing; inspect server storage.", 409)
                return {"ok": True, "storedPath": source.as_posix(), "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(), "duplicate": True, "session": active}
            if self._existing_matches(relative, body):
                return {"ok": True, "storedPath": relative.as_posix(), "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(), "duplicate": True,
                        "session": None, "sessionCompleted": True}
            result = self._save_bytes(staged, body)
            self._sync_directory((self.root / staged).parent)
            descriptor = {"format": "neural-qr-collection-session", "version": 1,
                          "id": str(uuid.uuid4()), "collectionCode": bundle["collectionCode"],
                          "senderSessionId": bundle["sessionId"], "profile": {key: bundle["profile"][key] for key in
                              ("size", "colors", "phases", "gridBlock", "permute", "pixelsPerCell", "holdMs", "parity")},
                          "datasetSplit": split, "created": bundle["created"]}
            self._write_session(descriptor)
            return {**result, "session": descriptor}

    def abandon_session(self, identifier):
        with self.session_lock:
            active = self._read_session()
            if not active or active["id"] != identifier:
                raise RecordingError("That collection is no longer active. Refresh the collector.", 409)
            self.session_path.unlink()
            self._sync_directory(self.root)
            return {"ok": True, "abandoned": True}

    def _finish_session(self, active):
        relative = self._sender_relative(active)
        final = self.root / relative
        staged = final.with_name(final.name + ".tmp")
        # Parent paths were derived from validated fields. Never follow a symlink.
        for path in (final, staged):
            if path.is_symlink() or self.root not in path.parent.resolve().parents or any(parent.is_symlink() for parent in path.parents if parent != self.root and self.root in parent.parents):
                raise RecordingError("Unsafe staged sender path.", 409)
        if staged.exists():
            try:
                os.link(staged, final)
            except FileExistsError:
                if not final.is_file() or final.read_bytes() != staged.read_bytes():
                    raise RecordingError("Final sender differs from staged data; neither file was removed.", 409)
            self._sync_directory(final.parent)
            staged.unlink()
            self._sync_directory(final.parent)
        elif not final.is_file():
            raise RecordingError("Staged sender is missing; receiver was saved, retry after restoring the sender.", 409)
        self.session_path.unlink()
        self._sync_directory(self.root)

    def save(self, bundle, split, body):
        relative = recording_destination(bundle, split)
        identifier = bundle.get("collectionSessionId")
        if identifier is None:
            # Already recorded files remain valid and never complete a managed session.
            return self._save_bytes(relative, body)
        if not isinstance(identifier, str) or bundle["format"] != "neural-qr-captures" or not bundle["records"]:
            raise RecordingError("Completing a collection requires receiver captures and its active session ID.")
        with self.session_lock:
            active = self._read_session()
            if active is None or active["id"] != identifier:
                # A response may have been lost after durable completion. Only an exact
                # existing receiver file may acknowledge storage of that old request.
                # Inactivity alone cannot prove sender finalization: the user may
                # have abandoned after an interrupted finalization.
                if self._existing_matches(relative, body):
                    return {"ok": True, "storedPath": relative.as_posix(), "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest(), "duplicate": True,
                            "sessionCompleted": False, "sessionInactive": True}
                raise RecordingError("This collection is no longer active. Your captures have not been changed.", 409)
            if (active["collectionCode"] != bundle["collectionCode"] or active["datasetSplit"] != split
                    or active["profile"] != bundle["profile"]):
                raise RecordingError("Receiver assignment disagrees with the active collection.", 409)
            for row in bundle["records"]:
                guard = row.get("stability", {})
                if (type(guard.get("guardMs")) is not int or not 500 <= guard["guardMs"] <= 5000
                        or not isinstance(guard.get("durationMs"), (int, float)) or guard["durationMs"] < guard["guardMs"]
                        or type(guard.get("observations")) is not int or guard["observations"] < 2):
                    raise RecordingError("Receiver capture has no phase settling evidence.")
            result = self._save_bytes(relative, body)
            self._sync_directory((self.root / relative).parent)
            self._finish_session(active)
            return {**result, "sessionCompleted": True}


class BenchHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript"}

    def __init__(self, *args, recordings=None, **kwargs):
        self.recordings = recordings
        super().__init__(*args, **kwargs)

    def json_response(self, status, value):
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def same_origin(self):
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return False
        origin = self.headers.get("Origin")
        scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
        return not origin or origin == f"{scheme}://{self.headers.get('Host', '')}"

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/recordings/config":
            if not self.same_origin():
                return self.json_response(403, {"error": "Use the collector's own server origin."})
            config = {"enabled": self.recordings is not None, "maxUploadBytes": self.recordings.max_bytes if self.recordings else MAX_UPLOAD_BYTES}
            if self.recordings:
                config["uploadToken"] = self.recordings.token
            return self.json_response(200, config)
        if path == "/api/collection-session":
            if not self.same_origin():
                return self.json_response(403, {"error": "Use the collector's own server origin."})
            try:
                return self.json_response(200, {"enabled": self.recordings is not None,
                    "session": self.recordings.current_session() if self.recordings else None})
            except RecordingError as error:
                return self.json_response(error.status, {"error": str(error)})
        if path.startswith("/api/"):
            return self.json_response(404, {"error": "Not found."})
        return super().do_GET()

    def do_POST(self):
        parsed = urlsplit(self.path)
        self.close_connection = True
        if parsed.path not in ("/api/recordings", "/api/collection-session", "/api/collection-session/abandon"):
            return self.json_response(404, {"error": "Not found."})
        if self.recordings is None:
            return self.json_response(404, {"error": "Uploads are disabled. Start the server with --recordings-dir."})
        if not self.same_origin() or not secrets.compare_digest(self.headers.get("X-Upload-Token", "").encode("utf-8"), self.recordings.token.encode("ascii")):
            return self.json_response(403, {"error": "Upload origin or token is invalid. Reload the collector."})
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            return self.json_response(415, {"error": "Upload a recording as application/json."})
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,12}", lengths[0]) or self.headers.get("Transfer-Encoding"):
            return self.json_response(411, {"error": "A single Content-Length is required."})
        length = int(lengths[0])
        limit = self.recordings.max_bytes if parsed.path == "/api/recordings" else min(MAX_SESSION_BYTES, self.recordings.max_bytes)
        if not 1 <= length <= limit:
            return self.json_response(413, {"error": "Recording exceeds the server upload limit."})
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/api/collection-session/abandon":
            valid_query = not query
        else:
            valid_query = set(query) == {"split"} and len(query["split"]) == 1 and query["split"][0] in SPLITS
        if not valid_query:
            return self.json_response(422, {"error": "Select exactly one split: train, validation or test."})
        if not self.recordings.slots.acquire(blocking=False):
            return self.json_response(429, {"error": "Two uploads are already running. Retry shortly."})
        try:
            self.connection.settimeout(60)
            body = self.rfile.read(length)
            if len(body) != length:
                raise RecordingError("Upload ended before the complete recording arrived.", 400)
            def invalid_constant(value):
                raise ValueError(f"Invalid JSON number: {value}")
            try:
                bundle = json.loads(body, parse_constant=invalid_constant)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise RecordingError("Recording must be valid UTF-8 JSON.", 400) from error
            if parsed.path == "/api/collection-session/abandon":
                if not isinstance(bundle, dict) or set(bundle) != {"id"} or not isinstance(bundle["id"], str):
                    raise RecordingError("Provide the active collection ID.")
                return self.json_response(200, self.recordings.abandon_session(bundle["id"]))
            operation = self.recordings.begin_session if parsed.path == "/api/collection-session" else self.recordings.save
            result = operation(bundle, query["split"][0], body)
            return self.json_response(200 if result["duplicate"] else 201, result)
        except RecordingError as error:
            return self.json_response(error.status, {"error": str(error)})
        except TimeoutError:
            return self.json_response(408, {"error": "Upload timed out; the recording remains on your device."})
        except OSError:
            return self.json_response(507, {"error": "Recording could not be saved. Check server storage and retry."})
        finally:
            self.recordings.slots.release()

    def send_head(self):
        # Static assets cannot expose uploaded JSON through an accidentally added symlink.
        path = Path(self.translate_path(self.path)).resolve()
        root = Path(self.directory).resolve()
        if path != root and root not in path.parents:
            self.send_error(404, "Not found")
            return None
        if self.recordings and (path == self.recordings.root or self.recordings.root in path.parents):
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=()")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "Directory listing is disabled")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true")
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--recordings-dir", type=Path, help="Enable collector uploads into this private directory outside dist")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if args.https and (args.cert is None or args.key is None):
        parser.error("--https requires --cert and --key; use a certificate trusted by the phone")
    if not args.https and (args.cert is not None or args.key is not None):
        parser.error("certificate arguments require --https")
    context = None
    if args.https:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.load_cert_chain(args.cert, args.key)
        except (OSError, ssl.SSLError) as exc:
            parser.error(str(exc))
    root = Path(__file__).resolve().parent / "dist"
    try:
        recordings = RecordingStore(args.recordings_dir, root) if args.recordings_dir else None
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer((args.host, args.port), partial(BenchHandler, directory=str(root), recordings=recordings))
    if context:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if context else "http"
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Optical Bench: {scheme}://{shown_host}:{args.port}", flush=True)
    if recordings:
        print(f"Recording uploads: {recordings.root} (train/validation/test, grouped by grid size; 800 MiB maximum per upload)", flush=True)
    if args.host == "0.0.0.0":
        print(f"For another device, replace localhost with this computer's LAN IP. Camera access requires trusted HTTPS.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
