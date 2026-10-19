# Third-party code

The browser apps bundle dependencies so camera transport and cryptography do
not require a CDN. Preserve their source, notices, manifests, and licenses.

| Dependency | Bundled version | License and provenance |
| --- | --- | --- |
| Nayuki QR Code generator | 1.8.0 | [MIT license](apps/qr-neural-key/dist/vendor/qrcodegen-LICENSE.txt), [manifest](apps/qr-neural-key/dist/vendor/manifest.json) |
| jsQR | 1.4.0 | [Apache-2.0 license](apps/qr-neural-key/dist/vendor/jsQR-LICENSE.txt), [provenance](apps/qr-neural-key/dist/vendor/README.md) |
| `@noble/hashes` SHAKE dependency closure | 2.4.0 | [MIT license](shared/browser-crypto/crypto/vendor/noble-hashes/LICENSE), [canonical provenance](shared/browser-crypto/crypto/vendor/noble-hashes/README.md) |

NumPy and PyTorch are Python dependencies installed separately; neither is vendored.
No project-wide license has been selected in this repository. These dependency
licenses apply to their respective code and do not select a license for the
project's original source.
