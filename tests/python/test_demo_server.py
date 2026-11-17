"""Static demo server configuration and path bounds; no sockets or TLS needed."""
from contextlib import redirect_stderr
import importlib.util
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_server", ROOT / "apps/neural-auth-demo/serve.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DemoServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.dist / "index.html").write_text("demo", encoding="utf-8")

    def test_configuration_has_local_default_and_explicit_https_credentials(self):
        _, args = MODULE.parse_args([])
        self.assertEqual((args.host, args.port, args.https), ("127.0.0.1", 8002, False))
        _, args = MODULE.parse_args(["--host", "0.0.0.0", "--https", "--cert", "cert.pem", "--key", "key.pem"])
        self.assertTrue(args.https)
        self.assertEqual(args.cert, Path("cert.pem"))
        for argv in (["--https"], ["--cert", "cert.pem"], ["--port", "0"], ["--port", "65536"]):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                MODULE.parse_args(argv)

    def test_static_paths_reject_traversal_and_external_targets(self):
        self.assertEqual(MODULE.asset_path(self.dist, "/?version=2"), self.dist / "index.html")
        self.assertEqual(MODULE.asset_path(self.dist, "/js/app.js?version=2"), self.dist / "js/app.js")
        for target in ("/../secret", "/%2e%2e/secret", "/js/../../secret", "/%5c..%5csecret", "/%00", "https://other.example/x", "//other.example/x"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                MODULE.asset_path(self.dist, target)

    def test_symlinked_files_directories_and_indexes_are_not_served(self):
        outside = self.root / "private"
        outside.mkdir()
        (outside / "secret").write_text("private", encoding="utf-8")
        (self.dist / "file").symlink_to(outside / "secret")
        (self.dist / "link").symlink_to(outside, target_is_directory=True)
        (self.dist / "nested").mkdir()
        (self.dist / "nested/index.html").symlink_to(self.dist / "index.html")
        for target in ("/file", "/link/secret", "/nested/"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                MODULE.asset_path(self.dist, target)

    def test_missing_assets_and_directory_lists_do_not_expose_other_files(self):
        handler = object.__new__(MODULE.DemoHandler)
        handler.directory = str(self.dist)
        handler.send_error = Mock()
        handler.path = "/missing.js"
        self.assertIsNone(handler.send_head())
        handler.send_error.assert_called_with(404, "Static asset not found")
        handler.path = "/../private"
        self.assertIsNone(handler.send_head())
        self.assertEqual(handler.send_error.call_args.args[0], 403)
        self.assertIsNone(handler.list_directory(str(self.dist)))
        handler.send_error.assert_called_with(403, "Directory listing is disabled")

    def test_runtime_mime_types_and_no_store_headers(self):
        handler = object.__new__(MODULE.DemoHandler)
        self.assertEqual(handler.guess_type("module.mjs"), "text/javascript")
        self.assertEqual(handler.guess_type("worker.js"), "text/javascript")
        self.assertEqual(handler.guess_type("runtime.wasm"), "application/wasm")
        handler.request_version = "HTTP/1.1"
        handler.wfile = BytesIO()
        handler.end_headers()
        self.assertIn(b"Cache-Control: no-store\r\n", handler.wfile.getvalue())
        self.assertIn(b"X-Content-Type-Options: nosniff\r\n", handler.wfile.getvalue())


if __name__ == "__main__":
    unittest.main()
