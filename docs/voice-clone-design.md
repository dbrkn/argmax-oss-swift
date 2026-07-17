# Voice Cloning in TTSKit — Design

Port of the ArgmaxPrototypes Qwen3-TTS voice-cloning capability (Python,
`argmax_prototypes/pipeline/tts/qwen3_tts/{voice_clone,mlx_voice_clone}.py` +
`pipeline._build_icl_embeddings`) into the production Swift `TTSKit`.

## What voice cloning adds

Custom-voice TTS selects a predefined speaker via an integer codec token
(`Qwen3Speaker.tokenID`) in the 7-slot codec prefix. Voice cloning replaces
that with information derived from a **reference clip** of the target speaker:

1. **x-vector**: an ECAPA-TDNN speaker embedding substituted into the speaker
   slot of the codec track (in place of the speaker token's embedding).
2. **ICL (in-context learning)**, the high-quality mode: additionally, the
   reference clip is encoded to 16-codebook RVQ codes (Mimi encoder) and the
   prefix is extended with a diagonal dual-track block — reference transcript +
   synthesis text on the text track over `CODEC_PAD`, then reference RVQ frame
   embeddings on the codec track under `TEXT_PAD` — so generation continues
   "in the voice of" the reference.

## Backend decision

Evaluated options for the reference-audio encoders (SpeakerEncoder,
SpeechEncoder, SpeechEncoderRVQ):

| Option | Verdict |
|---|---|
| **A. CoreML encoders** (assets exist: `qwen3_tts/{speaker_encoder,speech_encoder,speech_encoder_rvq}/12hz-0.6b-base/W16A16*`) | **Chosen.** Native to TTSKit (CoreML-only package, iOS-compatible), production-aligned, no new dependencies. Fixed compile-time windows (~10–15 s) cap the reference duration — acceptable: cloning references are conventionally 3–15 s, and the Python fixed-window path ships the same constraint. |
| B. x-vector-only sub-mode | **Included** (config flag, not a separate backend). Skips the Mimi/RVQ encoders entirely, supports prompt-cache reuse across utterances of the same voice; lower cloning fidelity. |
| C. Precomputed-prompt injection | **Included** (API accepts precomputed `speakerEmbedding` + `referenceCodes`). Zero-cost seam for parity testing against the Python encoders and for server-side setups that encode once and synthesize many times. |
| D. MLX-Swift encoders | **Included, as a Mac-only extension package** (`Extensions/TTSKitMLX`). The fixed CoreML window binds in the worst case — any reference longer than the compiled window (10/15 s) is silently truncated, degrading speaker identity — and worst case is the spec for a production capability. `mlx-swift` floors at macOS 14 / iOS 17 (no watchOS), so it cannot live inside `TTSKit` without a package-wide platform bump; the extension package keeps the core dependency-free. Cross-platform worst-case escape hatch: `argmax-cli tts --voice-clone-prompt prompt.json` accepts a precomputed `VoiceClonePrompt` from any encoder. |

Because of (C), "which backend" is also empirically testable end-to-end: the
OpenBench Phase-2 evals can run Swift-CoreML encoders vs Python-MLX encoders on
`voiceclone-eval` (SIM + WER) with everything else held fixed.

## Asset plan

The public `argmaxinc/ttskit-coreml` repo has **no voice-clone encoder assets**
today; they live in the private `argmaxinc/ttskit-internal` under
`qwen3_tts/{speaker_encoder,speech_encoder,speech_encoder_rvq}/12hz-0.6b-base/`.
Voice cloning also requires the **base**-family talker components
(`12hz-0.6b-base` version dir) rather than `-customvoice` — only base
checkpoints carry voice-clone-consistent weights.

- Development / internal evals: `--model-repo argmaxinc/ttskit-internal` (the
  config already supports a repo override).
- OSS release: publish the three encoder assets + `12hz-0.6b-base` component
  set to `argmaxinc/ttskit-coreml` (team decision, out of scope here).
- New `TTSModelVariant` case: `0.6b-base` → `12hz-0.6b-base`.

## Swift architecture

New files (all in `Sources/TTSKit/Qwen3TTS/VoiceClone/` unless noted):

| File | Contents |
|---|---|
| `MelSpectrogram.swift` | vDSP/Accelerate mel front-end matching the Python reference exactly: n_fft 1024, hop 256, win 1024 (hann), reflect-pad `(n_fft−hop)/2 = 384`, `center=False`, magnitude `sqrt(re²+im²+1e-9)`, 128-mel slaney filterbank (fmin 0, fmax 12000), `log(clamp(·, 1e-5))`. Filterbank generated at init (no bundled data files). |
| `Qwen3SpeakerEncoder.swift` | CoreML wrapper: mel `(1,128,1,T_mel)` fp16 → `speaker_embedding (1,D)`. Short references are **tiled** (not zero-padded) to the window so ECAPA pools over real speech; then mel right-pad/trim to the compile-time `T_mel`. |
| `Qwen3SpeechEncoder.swift` | CoreML wrapper: waveform `(1,1,1,L)` (right zero-pad/trim to the fixed window) → `projected_embeddings (1, 2·vqDim, 1, T_codes)`; channel split: `[..<vqDim]` semantic, `[vqDim...]` acoustic. |
| `Qwen3SpeechEncoderRVQ.swift` | FP32 per-codebook lookup: `(residual, codebookIdx) → (newResidual, codes)`. Two-branch loop: semantic branch codebook 0; acoustic branch **resets** the residual and runs codebooks 1–15. Output `(16, T_codes)` int32, then **padded-tail trim**: `validFrames = round(realSamples × T_codes / windowSamples)` — skipping this poisons the prefix with silence codes. |
| `VoiceCloneEncoder.swift` | Orchestrator + `VoiceClonePrompt` value type (`speakerEmbedding`, `referenceCodes?`, `referenceText?`). Also constructible directly from precomputed values (seam C). |
| `Qwen3GenerateTask+VoiceClone.swift` | ICL prefix assembly mirroring `_build_icl_embeddings`: 7-slot codec prefix with the x-vector substituted in the speaker slot; drop the trailing `codecBOS`; ICL block appended with the diagonal text/codec alignment; `rvqFrameEmbed` = codeEmbedder(code₀) + Σ multiCodeEmbedder(codeᵢ + (i−1)·2048). Prompt-fit validation against the CodeDecoder cache with a clear error (Python: error at full, warn under 128 slots of headroom). |
| (edit) `TTSKit.swift`, `Qwen3Config.swift`, `Models.swift` | `voiceToClone` input on generate options (`referenceAudio` URL or samples + `referenceText`, `xVectorOnly`), model loading for the three encoders (lazy — only when cloning requested), `0.6b-base` variant. |
| (edit) `ArgmaxCLI/TTSCLI.swift` | `--ref-audio`, `--ref-text`, `--x-vector-only`; cloning implied by `--ref-audio`. |
| `Extensions/TTSKitMLX/` (standalone package, own manifest) | Mac-only MLX-Swift ports of the variable-length encoders (backend D): `MlxVoiceCloneEncoder` (ECAPA `SpeakerEncoder` + Mimi `SpeechTokenizerEncoder`, no fixed reference window; 120 s memory-safety cap, see "Production impact") + `ttskit-mlx-cli encode` emitting `VoiceClonePrompt` JSON for `--voice-clone-prompt`. Not in the root manifests — mlx-swift floors at macOS 14. Parity-tested against Python MLX goldens (x-vector cosine > 0.999, ≥ 99% RVQ code match, 8 s + 32 s references). |

Reference audio loading: 24 kHz mono float32 via `AVAudioFile` +
`AVAudioConverter` (TTSKit currently has playback-only audio utilities; the
loader is new but small).

### Deliberate deviations from the Python reference (improvements)

- **Text chunking + ICL**: the Python CLI rebuilds the full ICL prefix per
  35-word text chunk (quadratic prefix work, and prosody resets at every chunk
  boundary). TTSKit already crossfades chunk audio; we keep per-chunk prefix
  rebuild for correctness, but the encoder outputs (`VoiceClonePrompt`) are
  computed **once** per generate call and reused across chunks — the Python
  version re-encodes the reference per chunk.
- **Prompt cache**: x-vector-only cloning gets prompt-cache support keyed by a
  hash of the speaker embedding (Python supports the equivalent
  `CacheInputs` path). ICL mode is uncacheable by construction (prefix embeds
  the utterance text); explicitly rejected with a clear reason instead of
  silently ignored.
- **Strict window accounting**: `frames_per_sample` tail-trim and the tiling
  rule are unit-tested against golden values exported from the Python
  implementation, since these were the empirically bug-prone spots.
- **No `--streaming` / `--no-chunk` compatibility flags**: TTSKit's
  streaming SpeechDecoder writer and sentence chunker subsume both.

## Parity & testing

1. **Golden-value unit tests**: a Python export script
   (`scripts/export_voice_clone_goldens.py`, runs in the ArgmaxPrototypes venv)
   dumps, for a fixed 8 s reference: the mel matrix, tiled/padded windows,
   x-vector, RVQ codes pre/post trim, and the first/last ICL prefix embedding
   vectors. Swift unit tests assert against these within fp16 tolerances.
2. **Integration test**: end-to-end clone of a bundled short reference,
   gated on asset availability (models-path override), asserting audio
   duration/energy sanity — mirrors `TTSKitIntegrationTests` conventions.
3. **End-to-end eval (Phase 2)**: OpenBench `argmax-speech-generation-oss`
   pipeline gains voice-clone mode (ref audio/text pass-through identical to
   the prototype pipeline), run on `voiceclone-eval` (374 pairs, held-out
   `target_audio` SIM yardstick + WavLM ASV checkpoint) and compared against
   the Python `tts-cli` numbers on the same dataset.

## Phasing

1. Mel + encoders + goldens (buildable & unit-tested without talker changes).
2. ICL prefix + generate-task integration + CLI.
3. Local A/B vs Python `tts-cli` on a handful of `voiceclone-eval` samples.
4. OpenBench Phase 2 (separate branch on the OpenBench fork + internal
   workflow input `pipeline=argmax-speech-generation-oss`).

## Production impact: MLX vs CoreML encoders (measured)

Measured on an M-series Mac (24 kHz mono references; encoders only — the
talker/vocoder stay CoreML in both configurations):

| Dimension | CoreML (W16A16, 10 s window, ANE) | MLX (fp32, GPU/Metal) |
|---|---|---|
| Encode latency | ~158 ms constant (fixed window) | ~5.4 ms/s of reference: 23 ms @3 s, 47 ms @8 s, 173 ms @32 s |
| Reference > window | Silently truncated | Full length |
| Load time | 41 s cold (first ANE compile), 4.8 s warm | 2.1 s |
| Process RSS after load | +638 MB | +585 MB |
| Peak accelerator memory | ~flat (window-bounded) | Scales ~90 MB per reference second (0.7 GB @3 s → 2.9 GB @32 s) |
| Weights on disk | 124 MB | ~2.0 GB as published (only the speaker-encoder slice of the 1.3 GB talker file is used) |
| Compute placement / power | ANE, ~2–5 W, leaves GPU free | GPU, ~15–40 W, contends with other Metal work |
| Precision | fp16 | fp32 (matches the Python research reference) |

Consequences:

1. **Reference-length cap is a pre-ship requirement for the MLX path**: peak
   memory grows unbounded with reference length (~27 GB for a 5-minute clip).
   Cap (~120 s) with a clear error, or window the Mimi encode with left
   context.
2. **Per-process weights (~600 MB) are not page-shared** the way `mlmodelc`
   mmaps are — multi-worker servers should route cloning through a single
   encode service.
3. **Pruned checkpoint before shipping**: publish an encoder-only safetensors
   set (~750 MB) instead of the full 2 GB talker+tokenizer download.
4. Placement is complementary (encode on GPU, generate on ANE); routing can be
   automatic — CoreML for ≤10 s references / battery deployments, MLX beyond —
   since both emit the identical `VoiceClonePrompt`.
