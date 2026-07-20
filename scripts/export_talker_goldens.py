# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Export golden values from the Python MLX Qwen3-TTS talker (mlx-audio).

Consumed by `Extensions/TTSKitMLX/Tests/TTSKitMLXTests/TalkerParityTests.swift`
to prove the MLX-Swift `MlxCodeDecoder` port matches the Python MLX talker:
batched prefill last-position logits/hidden within fp tolerance, and 20 greedy
decode steps token-exact.

Run from the ArgmaxPrototypes venv (has mlx + mlx-audio):

    /path/to/ArgmaxPrototypes/.venv/bin/python scripts/export_talker_goldens.py \
      --output-dir Tests/TTSKitTests/Resources/VoiceCloneGoldens

The prefix mirrors a production generation prompt: the codec control block
(THINK / THINK_BOS / language / THINK_EOS / PAD / PAD / BOS) followed by
pseudo-random audio codes, each summed with the projected text-PAD embedding —
all embeddings taken from the talker's own tables so both sides compute the
identical feedback path. Everything is cast to fp16 before entering the
transformer, matching the fp16 embeddings TTSKit supplies at runtime.

Outputs (all float32 little-endian, values pre-rounded to fp16):
    mlx_talker_prefix_embeds.bin        (L, 1024) prefill input embeddings
    mlx_talker_prefill_logits.bin       (3072,) last-position prefill logits
    mlx_talker_prefill_hidden.bin       (1024,) last-position prefill hidden
    mlx_talker_feedback_text_embed.bin  (1024,) projected text-PAD embedding
    mlx_talker_goldens.json             prefix ids, greedy tokens, shapes
"""

import argparse
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def resolve_snapshot(repo_id: str) -> str:
    pattern = (
        str(Path.home())
        + f"/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots/*"
    )
    candidates = [p for p in sorted(glob.glob(pattern)) if Path(p, "config.json").exists()]
    if not candidates:
        raise SystemExit(f"No cached snapshot of {repo_id}; download it first.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-prefix-codes", type=int, default=40)
    parser.add_argument("--greedy-steps", type=int, default=20)
    args = parser.parse_args()

    from mlx_audio.tts.utils import load_model

    snapshot = resolve_snapshot(args.repo_id)
    model = load_model(snapshot)
    talker = model.talker
    cfg = model.config.talker_config

    codec_embedding = talker.model.codec_embedding
    text_embedding = talker.model.text_embedding
    text_projection = talker.text_projection

    # Projected text-PAD embedding: the text-track filler under every codec
    # position past the prompt text (fp32 out of the quantized ResizeMLP).
    tts_pad_id = 151671
    text_pad_proj = text_projection(text_embedding(mx.array([tts_pad_id])))[0]

    # Codec track: control block + deterministic pseudo-random audio codes.
    control_ids = [
        cfg.codec_think_id,  # 2154
        cfg.codec_think_bos_id,  # 2156
        cfg.codec_language_id["english"],  # 2050
        cfg.codec_think_eos_id,  # 2157
        cfg.codec_pad_id,  # 2148 (speaker slot: base model, no speaker token)
        cfg.codec_pad_id,  # 2148
        cfg.codec_bos_id,  # 2149
    ]
    audio_ids = [(i * 97 + 13) % 2048 for i in range(args.audio_prefix_codes)]
    prefix_ids = control_ids + audio_ids

    # Each prefix position: codec embedding + projected text PAD, cast fp16 —
    # the dtype TTSKit's [FloatType] embeddings arrive in.
    prefix = (codec_embedding(mx.array(prefix_ids)) + text_pad_proj).astype(mx.float16)

    cache = talker.make_cache()
    logits, hidden = talker(prefix[None], cache=cache)
    mx.eval(logits, hidden)
    prefill_logits = np.array(logits[0, -1].astype(mx.float32))
    prefill_hidden = np.array(hidden[0, -1].astype(mx.float32))

    tokens = []
    current = int(mx.argmax(logits[0, -1]).item())
    for _ in range(args.greedy_steps):
        tokens.append(current)
        step_embed = (codec_embedding(mx.array([current])) + text_pad_proj).astype(mx.float16)
        logits, hidden = talker(step_embed[None], cache=cache)
        mx.eval(logits)
        current = int(mx.argmax(logits[0, -1]).item())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.array(prefix.astype(mx.float32)).astype(np.float32).tofile(out / "mlx_talker_prefix_embeds.bin")
    prefill_logits.astype(np.float32).tofile(out / "mlx_talker_prefill_logits.bin")
    prefill_hidden.astype(np.float32).tofile(out / "mlx_talker_prefill_hidden.bin")
    np.array(text_pad_proj.astype(mx.float32)).astype(np.float32).tofile(
        out / "mlx_talker_feedback_text_embed.bin"
    )

    goldens = {
        "repoId": args.repo_id,
        "snapshot": Path(snapshot).name,
        "prefixIds": prefix_ids,
        "prefixLength": len(prefix_ids),
        "hiddenSize": cfg.hidden_size,
        "vocabSize": cfg.vocab_size,
        "greedyTokens": tokens,
        "prefillLogitsFirst8": [float(v) for v in prefill_logits[:8]],
        "prefillArgmax": tokens[0],
    }
    (out / "mlx_talker_goldens.json").write_text(json.dumps(goldens, indent=2) + "\n")

    print(f"prefix: {len(prefix_ids)} tokens, greedy tokens: {tokens}")
    print(f"wrote goldens to {out}")


if __name__ == "__main__":
    main()
