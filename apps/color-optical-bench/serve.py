#!/usr/bin/env python3
"""Serve the static optical bench. Python 3.10+, standard library only.

Local webcam: python serve.py
Phone/LAN:   python serve.py --host 0.0.0.0 --https --cert certs/dev.pem --key certs/dev-key.pem
The phone must trust the certificate. This server performs no decoding or uploads.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl


class BenchHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript"}

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
    server = ThreadingHTTPServer((args.host, args.port), partial(BenchHandler, directory=str(root)))
    if context:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    scheme = "https" if context else "http"
    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Optical Bench: {scheme}://{shown_host}:{args.port}", flush=True)
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
