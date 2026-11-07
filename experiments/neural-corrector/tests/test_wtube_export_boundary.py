"""Do not label a training-only W-tube checkpoint as a supported browser bundle."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_EXPORT = all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx"))
if HAS_EXPORT:
    from export import export_smoke


@unittest.skipUnless(HAS_EXPORT, "PyTorch and ONNX required")
class WTubeExportBoundaryTests(unittest.TestCase):
    def test_explicit_error_before_creating_or_replacing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"bundle"
            with self.assertRaisesRegex(ValueError, "browser bundle export"):
                export_smoke(out, profile="W-tube", sizes=(16,), check_ort=False)
            self.assertFalse(out.exists())
            out.mkdir()
            sentinel = out/"keep.txt"
            sentinel.write_text("previous bundle")
            with self.assertRaisesRegex(ValueError, "browser bundle export"):
                export_smoke(out, profile="W-tube", sizes=(64,), check_ort=False)
            self.assertEqual(sentinel.read_text(), "previous bundle")
            self.assertEqual(list(out.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
