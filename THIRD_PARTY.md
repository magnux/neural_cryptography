# Third-party code

The browser apps bundle dependencies so camera transport and cryptography do
not require a CDN. Preserve their source, notices, manifests, and licenses.

| Dependency | Bundled version | License and provenance |
| --- | --- | --- |
| Nayuki QR Code generator | 1.8.0 | [MIT license](apps/qr-neural-key/dist/vendor/qrcodegen-LICENSE.txt), [manifest](apps/qr-neural-key/dist/vendor/manifest.json) |
| jsQR | 1.4.0 | [Apache-2.0 license](apps/qr-neural-key/dist/vendor/jsQR-LICENSE.txt), [provenance](apps/qr-neural-key/dist/vendor/README.md) |
| `@noble/hashes` SHAKE dependency closure | 2.4.0 | [MIT license](shared/browser-crypto/crypto/vendor/noble-hashes/LICENSE), [canonical provenance](shared/browser-crypto/crypto/vendor/noble-hashes/README.md) |
| ONNX Runtime Web (optional Corrector Lab) | 1.23.2 | [MIT license](experiments/neural-corrector/browser-runtime/LICENSE.onnxruntime), [third-party notices](experiments/neural-corrector/browser-runtime/ThirdPartyNotices.onnxruntime.txt), [pinned package](experiments/neural-corrector/browser-runtime/package.json) |

The Corrector Lab runtime is installed separately using its npm lockfile and copied
into ignored app-local assets by `npm run prepare:corrector`, along with both
notices. Its generated runtime manifest records SHA-256 hashes. No CDN is used
at inference time. These files are separate from the existing transport and
crypto dependencies; ordinary `prepare:apps` does not install them.

NumPy and PyTorch are Python dependencies installed separately; neither is vendored.
The project's original source and documentation use [Apache-2.0](LICENSE), with
project/author attribution in [NOTICE](NOTICE). The dependency licenses above
apply to their respective code and remain unchanged. The project license does
not replace third-party licenses or grant rights in externally supplied data or
checkpoints; see [licensing scope](docs/LICENSING.md).

