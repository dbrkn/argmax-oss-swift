//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
import MLX

// MARK: - Errors

/// Errors thrown by the MLX voice-clone encoders.
public enum TTSKitMLXError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidCheckpoint(String)
    case invalidInput(String)
    case referenceTooLong(String)

    public var errorDescription: String? {
        switch self {
            case let .modelNotFound(message),
                let .invalidCheckpoint(message),
                let .invalidInput(message),
                let .referenceTooLong(message):
                return message
        }
    }
}

// MARK: - Checkpoint snapshot resolution

/// Locates a local Qwen3-TTS Base-family MLX checkpoint snapshot.
///
/// The encoders need two pieces of a Base-family checkpoint (CustomVoice repos
/// ship the talker only): the root `config.json` + `model*.safetensors` (ECAPA
/// speaker encoder; shares its safetensors file with the talker) and the
/// `speech_tokenizer/` subfolder (Mimi encoder). The encoder weights are
/// unquantized (fp32/bf16) even in the 8-bit repos — mlx-audio skips
/// quantizing `speaker_encoder` / `speech_tokenizer` — so no quantized-layer
/// support is needed.
///
/// This package intentionally does not download from the Hub; pass an explicit
/// directory or pre-fetch the snapshot (e.g. with `huggingface-cli download`).
public enum ModelDirectory {
    /// Default checkpoint repo: the only mlx-community family that includes
    /// the ECAPA speaker-encoder weights.
    public static let defaultRepoID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

    /// Resolve the default repo's cached Hugging Face snapshot, mirroring the
    /// `~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{rev}/` layout.
    public static func defaultSnapshot() throws -> URL {
        let cacheRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--" + defaultRepoID.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let candidates = (try? FileManager.default.contentsOfDirectory(
            at: cacheRoot, includingPropertiesForKeys: nil
        )) ?? []
        for candidate in candidates.sorted(by: { $0.path < $1.path }) {
            if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("config.json").path) {
                return candidate
            }
        }
        throw TTSKitMLXError.modelNotFound(
            "No cached snapshot of \(defaultRepoID) under \(cacheRoot.path). "
                + "Download it first (huggingface-cli download \(defaultRepoID)) or pass --model-dir."
        )
    }

    /// Load and merge every matching safetensors file in `directory`.
    static func loadWeights(directory: URL, glob prefix: String) throws -> [String: MLXArray] {
        let files = ((try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )) ?? [])
        .filter { $0.lastPathComponent.hasPrefix(prefix) && $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw TTSKitMLXError.modelNotFound(
                "No \(prefix)*.safetensors found in \(directory.path)"
            )
        }
        var weights = [String: MLXArray]()
        for file in files {
            for (key, value) in try MLX.loadArrays(url: file) {
                weights[key] = value
            }
        }
        return weights
    }
}

// MARK: - Weight dictionary access

/// Throwing accessor over a sanitized `[key: MLXArray]` checkpoint dictionary.
struct Checkpoint {
    let weights: [String: MLXArray]
    let source: URL

    func tensor(_ key: String) throws -> MLXArray {
        guard let value = weights[key] else {
            throw TTSKitMLXError.invalidCheckpoint("Missing weight '\(key)' in \(source.path)")
        }
        return value
    }
}

// MARK: - Weight layout heuristic

/// Whether a rank-3 conv weight is already in the MLX `(out, kernel, in)`
/// layout rather than PyTorch's `(out, in, kernel)`.
///
/// Faithful port of the Python reference's `check_array_shape_qwen3`: needed
/// because mlx-community checkpoints store the speaker-encoder convs already
/// transposed while raw exports keep the PyTorch layout.
func isMLXConvLayout(_ shape: [Int]) -> Bool {
    guard shape.count == 3 else { return false }
    let (dim2, dim3) = (shape[1], shape[2])
    if dim2 == 1 {
        // (out, 1, dim3): a large dim3 is in_channels (MLX), a small one is kernel (PyTorch).
        return dim3 > 64
    }
    if dim3 == 1 {
        // (out, dim2, 1): a large dim2 is in_channels (PyTorch), a small one is kernel (MLX).
        return dim2 <= 64
    }
    // General heuristic: kernel_size < in_channels is the common case.
    return dim2 < dim3
}

/// Return a rank-3 conv weight in the MLX `(out, kernel, in)` layout,
/// transposing from PyTorch's `(out, in, kernel)` when needed.
func asMLXConvWeight(_ weight: MLXArray) -> MLXArray {
    guard weight.ndim == 3, !isMLXConvLayout(weight.shape) else { return weight }
    return weight.transposed(0, 2, 1)
}

// MARK: - JSON config decoding

/// Decode `container[key]` when present, else the dataclass default — the
/// Swift analogue of the Python reference's `filter_dict_for_dataclass`
/// (unknown JSON keys are ignored by the decoder; missing keys fall back).
extension KeyedDecodingContainer {
    func value<T: Decodable>(_ type: T.Type, forKey key: Key, default defaultValue: T) -> T {
        (try? decodeIfPresent(type, forKey: key)) ?? defaultValue
    }
}

#endif // canImport(MLX)
