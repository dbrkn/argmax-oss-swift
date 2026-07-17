# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Export golden values from the Python voice-clone reference implementation.

Run from an ArgmaxPrototypes checkout (its venv has numpy/librosa/torch):

    PYTHONPATH=/path/to/ArgmaxPrototypes \
      python scripts/export_voice_clone_goldens.py --ref-audio ref.wav \
      --output-dir Tests/TTSKitTests/Resources/VoiceCloneGoldens

Outputs (consumed by TTSKit unit tests):
    ref_audio.bin           float32 mono 24 kHz samples of the reference
    mel.bin                 float32 (128, T) mel matrix of the raw reference
    speaker_mel_input.bin   float32 (128, 960) tiled+padded SpeakerEncoder input
    speech_window.bin       float32 (240000,) padded SpeechEncoder input window
    goldens.json            shapes, tiling counts, valid-frame trim, mel params
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mel-frames", type=int, default=960, help="SpeakerEncoder mel window (frames)")
    parser.add_argument("--speech-window", type=int, default=240000, help="SpeechEncoder waveform window (samples)")
    parser.add_argument("--num-codes", type=int, default=125, help="SpeechEncoder output frames")
    args = parser.parse_args()

    from argmax_prototypes.pipeline.tts.qwen3_tts.voice_clone import (
        MelConfig,
        load_reference_audio,
        mel_spectrogram,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    waveform = load_reference_audio(args.ref_audio)  # 24 kHz mono float32
    cfg = MelConfig()

    # Raw mel of the full reference (no windowing) — validates the DSP front-end.
    mel = mel_spectrogram(waveform, cfg)  # (1, 128, T)
    mel2d = np.asarray(mel[0], dtype=np.float32)

    # SpeakerEncoder input prep: tile short refs to the mel window, then
    # right-pad/trim the mel to exactly mel_frames.
    hop = cfg.hop_size
    window_samples = args.mel_frames * hop
    tiles = 1
    tiled = waveform
    while len(tiled) < window_samples:
        tiled = np.concatenate([tiled, waveform])
        tiles += 1
    tiled = tiled[:window_samples]
    speaker_mel = np.asarray(mel_spectrogram(tiled, cfg)[0], dtype=np.float32)
    if speaker_mel.shape[1] < args.mel_frames:
        speaker_mel = np.pad(speaker_mel, ((0, 0), (0, args.mel_frames - speaker_mel.shape[1])))
    else:
        speaker_mel = speaker_mel[:, : args.mel_frames]

    # SpeechEncoder input prep: right zero-pad/trim the waveform.
    speech_window = np.zeros(args.speech_window, dtype=np.float32)
    n = min(len(waveform), args.speech_window)
    speech_window[:n] = waveform[:n]

    # RVQ padded-tail trim: frames_per_sample bookkeeping.
    frames_per_sample = args.num_codes / args.speech_window
    valid_frames = int(round(len(waveform) * frames_per_sample))

    waveform.astype(np.float32).tofile(out / "ref_audio.bin")
    mel2d.tofile(out / "mel.bin")
    speaker_mel.tofile(out / "speaker_mel_input.bin")
    speech_window.tofile(out / "speech_window.bin")

    goldens = {
        "melParams": {
            "nFFT": cfg.n_fft,
            "numMels": cfg.num_mels,
            "hopSize": cfg.hop_size,
            "winSize": cfg.win_size,
            "fmin": cfg.fmin,
            "fmax": cfg.fmax,
            "samplingRate": cfg.sampling_rate,
        },
        "refSamples": int(len(waveform)),
        "melShape": list(mel2d.shape),
        "melChecksumRowMeans": [float(x) for x in mel2d.mean(axis=1)[:8]],
        "melFirstFrame": [float(x) for x in mel2d[:8, 0]],
        "melLastFrame": [float(x) for x in mel2d[:8, -1]],
        "speakerMelShape": list(speaker_mel.shape),
        "speakerTiles": tiles,
        "speechWindowSamples": args.speech_window,
        "numCodes": args.num_codes,
        "validFrames": valid_frames,
    }
    (out / "goldens.json").write_text(json.dumps(goldens, indent=1))
    print(json.dumps(goldens, indent=1))


if __name__ == "__main__":
    main()
