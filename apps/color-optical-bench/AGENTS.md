# Optical Bench development

This is a preliminary experiment, not a deployed cryptographic protocol.
Maintain one current wire format and one content representation. Do not add
backward compatibility, old-format readers/writers, automatic migrations, or
fallback parsing for previous bench releases. Update both devices together.

Keep processing in the browser and retain independent live/lab state. Diagnostic
ground truth may measure errors only after classification; it must never help
decode or repair payloads. Calibration references are measurements, not payload
truth. CRC checks remain mandatory. Preserve bounded memory and worker execution.
