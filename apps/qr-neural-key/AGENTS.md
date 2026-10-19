# QR Neural Key POC

Browser-local QR transport and neural ZKBoo authentication experiment.
Use one current wire format per layer; no compatibility readers or migrations.
Bundle dependencies locally with licenses. Use standard QR error correction.
Keep buffers, retries, queues, and decoded input bounded. Transport integrity
is not authentication. Preserve the transport lab while integrating the compact
v2 proof engine in a separate authentication view. Secrets stay inside a worker;
no keys or payloads go to a server. Authentication uses pinned public identities,
a fresh exact verifier challenge, verifier-controlled repetition policy, explicit
pairing confirmation, expiry, and replay rejection. Do not equate the POC with
an audited cryptographic system or treat optical proximity as secrecy.
