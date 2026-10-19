# Vendored SHAKE implementation

These unmodified ESM files are the SHA-3/SHAKE dependency closure from
`@noble/hashes` **2.4.0**, distributed under the included MIT `LICENSE`.
Only SHAKE-256 is used by this application. There is no runtime npm, CDN,
network fetch, or Node dependency.

Sources:
- https://www.npmjs.com/package/@noble/hashes/v/2.4.0
- https://github.com/paulmillr/noble-hashes/tree/2.4.0
- https://registry.npmjs.org/@noble/hashes/-/hashes-2.4.0.tgz

The npm tarball was downloaded into an external staging directory using
`npm pack @noble/hashes@2.4.0`. Its SHA-512 digest was checked against the
registry integrity before copying these four files, with no source edits.

Tarball integrity:

```
sha512-X5XaVWZIBCT7HHZGm5I7ZQXDwLG+bGXuSrMQAW+7Zvl87h1kmc1ZB1VSRJcpUfoUrGQp4Fkoxm5kZ+Ms+aW+eA==
```

SHA-256 of copied files:

```
9a81e1edb24eae27b335533220167609cfb58008c5690e140ce478acdc669f32  sha3.js
037ad49adb78168b6b699598fd33f85f7877456b1d28b4d032e2a1a16807947c  utils.js
b09da8c07fe8187c07649494cdb7cd0bcf13df90b506a9473d19e4d5f8c2e102  _u64.js
4f221aee6e072336700c408c68ab3b96a3fc09f6aebe6f48f1bd99e5ef13faec  LICENSE

```

This provenance check and the application tests do not constitute a security
audit of this package or of the experimental proof protocol.
