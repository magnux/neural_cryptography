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
from urllib.parse import parse_qs, urlsplit


MAX_UPLOAD_BYTES = 160 * 1024 * 1024
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

    def save(self, bundle, split, body):
        relative = recording_destination(bundle, split)
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
        if path.startswith("/api/"):
            return self.json_response(404, {"error": "Not found."})
        return super().do_GET()

    def do_POST(self):
        parsed = urlsplit(self.path)
        self.close_connection = True
        if parsed.path != "/api/recordings":
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
        if not 1 <= length <= self.recordings.max_bytes:
            return self.json_response(413, {"error": "Recording exceeds the server upload limit."})
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) != {"split"} or len(query["split"]) != 1 or query["split"][0] not in SPLITS:
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
                bundle = json.loads(body.decode("utf-8"), parse_constant=invalid_constant)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise RecordingError("Recording must be valid UTF-8 JSON.", 400) from error
            result = self.recordings.save(bundle, query["split"][0], body)
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
        print(f"Recording uploads: {recordings.root} (train/validation/test, grouped by grid size; 160 MiB per upload)", flush=True)
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
