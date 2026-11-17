#!/usr/bin/env python3
"""Serve only the prepared neural-auth-demo/dist directory. Standard library only.

Local webcam: python apps/neural-auth-demo/serve.py
Phone/LAN: python apps/neural-auth-demo/serve.py --host 0.0.0.0 --https \
    --cert certs/dev.pem --key certs/dev-key.pem
Use a certificate trusted by the phone and matching the server hostname.
There are no upload, authentication, inference or other application APIs.
"""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl
from urllib.parse import unquote, urlsplit


DIST = Path(__file__).resolve().parent / "dist"


def asset_path(directory, request_target):
    """Resolve a static request without traversal, symlinks or directory listings."""
    root = Path(directory).resolve()
    target = urlsplit(request_target)
    if target.scheme or target.netloc:
        raise ValueError("Only local asset paths are served.")
    decoded = unquote(target.path, errors="strict")
    if "\0" in decoded or "\\" in decoded:
        raise ValueError("Invalid asset path.")
    parts = decoded.lstrip("/").split("/")
    if any(part in (".", "..") for part in parts):
        raise ValueError("Path traversal is not allowed.")
    path = root.joinpath(*parts)
    if path.is_dir():
        path /= "index.html"
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlinked assets are not served.")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Asset must stay inside dist.")
    return path


class DemoHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript",
        ".wasm": "application/wasm", ".onnx": "application/octet-stream",
        ".f32": "application/octet-stream",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def translate_path(self, path):
        return str(asset_path(self.directory, path))

    def send_head(self):
        try:
            path = asset_path(self.directory, self.path)
        except (ValueError, UnicodeError):
            self.send_error(403, "Asset path is outside the served app or contains a symlink")
            return None
        if not path.is_file():
            self.send_error(404, "Static asset not found")
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(403, "Directory listing is disabled")
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--https", action="store_true")
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.https and (not args.cert or not args.key):
        parser.error("--https requires both --cert and --key")
    if not args.https and (args.cert or args.key):
        parser.error("Use --https with --cert and --key")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    if DIST.is_symlink() or not (DIST / "index.html").is_file():
        parser.error("Expected this app's dist/index.html in a real dist directory")
    context = None
    if args.https:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(args.cert, args.key)
        except (OSError, ssl.SSLError) as error:
            parser.error(f"Cannot load the HTTPS certificate/key: {error}")
    with ThreadingHTTPServer((args.host, args.port), DemoHandler) as server:
        if context:
            server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https" if context else "http"
        print(f"Serving only {DIST} at {scheme}://{args.host}:{args.port}/", flush=True)
        if context:
            print("Open the certificate's hostname on both devices; each device must trust it.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
