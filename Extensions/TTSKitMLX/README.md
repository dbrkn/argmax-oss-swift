# TTSKitMLX

Mac-only MLX-Swift ports of the Qwen3-TTS voice-clone encoders, packaged as a
standalone SwiftPM extension to [TTSKit](../../README.md).

## Why this package exists

TTSKit's built-in voice-clone encoders are CoreML assets with
**compile-time-fixed input windows** (~10–15 s): shorter references are
tiled/padded and longer ones are **silently truncated**, which caps the usable
reference duration and degrades speaker identity for long clips. The MLX
encoders are variable-length by construction — the ECAPA speaker encoder pools
statistics over however many mel frames it is given, and the Mimi encoder is a
causal conv/transformer stack — so any reference duration works.

It is a separate package (not a `TTSKit` target) because of platform floors:
`mlx-swift` requires macOS 14+ / iOS 17+ and has no watchOS support, while the
root package floors at macOS 13 / watchOS 10. Keeping the MLX dependency out
of the root manifests keeps `TTSKit` dependency-free. See
[docs/voice-clone-design.md](../../docs/voice-clone-design.md), "Backend
decision" row D.

## Contents

- **`TTSKitMLX` library** — `MlxVoiceCloneEncoder`, producing TTSKit's
  `VoiceClonePrompt` from a 24 kHz mono reference waveform:
  - `SpeakerEncoder`: ECAPA-TDNN x-vector (1024-d), full-length mel input.
  - `SpeechTokenizerEncoder`: Mimi-style SeaNet + transformer + split-RVQ
    encoder producing 16-codebook reference codes at 12.5 Hz.
- **`ttskit-mlx-cli` executable** — `encode` subcommand that writes the
  prompt JSON.

Weights load from a Base-family Qwen3-TTS MLX checkpoint snapshot (default:
the cached Hugging Face snapshot of
`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`; the encoder weights are
unquantized even in the 8-bit repos). This package does not download from the
Hub — pre-fetch the snapshot or pass `--model-dir`.

## Encode → synthesize workflow

Encoding and synthesis are decoupled through the `VoiceClonePrompt` JSON, so
the Mac-only MLX encoders can serve any platform's synthesis:

```shell
# 1. Encode the reference once (any length up to the memory cap):
swift run -c release ttskit-mlx-cli encode \
    --ref-audio reference.wav \
    --ref-text "transcript of the reference clip" \
    --output prompt.json

# 2. Synthesize with the main CLI (CoreML talker, any supported platform):
argmax-cli tts --voice-clone-prompt prompt.json --text "..." --output-path out.wav
```

`--x-vector-only` skips the RVQ codes for lower-fidelity but
prompt-cacheable cloning; `--raw-float32` accepts headerless little-endian
float32 samples (24 kHz mono) instead of an audio container.

## Reference-length cap

`MlxVoiceCloneEncoder(maxReferenceSeconds: 120)` (CLI:
`--max-reference-seconds`). Encode peak Metal memory scales roughly **90 MB
per reference second** (measured: 0.7 GB @ 3 s → 2.9 GB @ 32 s — see the
design doc's "Production impact" section), so an unbounded reference would
exhaust GPU memory (~27 GB for a 5-minute clip). References over the cap
**throw** rather than silently truncate — truncation is exactly the CoreML
failure mode this package exists to remove. Raise the cap deliberately on
machines with enough GPU memory.

## Building and testing

`mlx-swift`'s Metal shaders cannot be compiled by command-line SwiftPM, so
plain `swift build` produces binaries that fail at runtime with
`Failed to load the default metallib`. Use `xcodebuild` (or Xcode):

```shell
# Tests (includes Python-parity goldens; needs the cached HF snapshot):
xcodebuild test -scheme TTSKitMLX-Package -destination 'platform=macOS' \
    -derivedDataPath .build/xcode
```

To use plain `swift test` / `swift run` after that first `xcodebuild` pass,
graft the shader bundle it produced onto the SwiftPM products:

```shell
# swift test:
mkdir -p .build/arm64-apple-macosx/debug/TTSKitMLXPackageTests.xctest/Contents/Resources
cp -R .build/xcode/Build/Products/Debug/mlx-swift_Cmlx.bundle \
    .build/arm64-apple-macosx/debug/TTSKitMLXPackageTests.xctest/Contents/Resources/
swift test

# release CLI:
swift build -c release
cp -R .build/xcode/Build/Products/Debug/mlx-swift_Cmlx.bundle .build/arm64-apple-macosx/release/
.build/arm64-apple-macosx/release/ttskit-mlx-cli encode ...
```

The parity tests assert against goldens exported from the Python reference
(`Tests/TTSKitTests/Resources/VoiceCloneGoldens/`): x-vector cosine > 0.999
and ≥ 99% exact RVQ-code match on both an 8 s and a 32 s reference — the 32 s
case (403 code frames) is past the CoreML window and is the variable-length
proof.
