//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation
import TTSKit
import TTSKitMLX

@main
struct TTSKitMLXCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ttskit-mlx-cli",
        abstract: "MLX voice-clone encoders for TTSKit (Mac-only)",
        discussion: """
        Encodes reference audio of any length (up to the memory-safety cap) into a \
        VoiceClonePrompt JSON that `argmax-cli tts --voice-clone-prompt` consumes, \
        bypassing the CoreML encoders' fixed reference window.
        """,
        subcommands: [EncodeCLI.self, BenchCLI.self]
    )
}

struct EncodeCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "encode",
        abstract: "Encode a reference clip into a VoiceClonePrompt JSON"
    )

    @Option(help: "Path to the reference audio (any AVAudioFile-readable format; resampled to 24 kHz mono)")
    var refAudio: String

    @Option(help: "Transcript of the reference clip (required for ICL cloning; stored in the prompt)")
    var refText: String?

    @Flag(help: "Skip the RVQ reference codes; x-vector-only prompts are lower fidelity but cacheable")
    var xVectorOnly: Bool = false

    @Flag(help: "Treat --ref-audio as headerless little-endian float32 samples at 24 kHz mono")
    var rawFloat32: Bool = false

    @Option(help: """
    Qwen3-TTS Base-family MLX checkpoint snapshot directory \
    (default: the cached Hugging Face snapshot of \(ModelDirectory.defaultRepoID))
    """)
    var modelDir: String?

    @Option(help: """
    Reference-length cap in seconds. Encode peak Metal memory scales ~90 MB per \
    reference second; raise deliberately on machines with enough GPU memory.
    """)
    var maxReferenceSeconds: Double = 120

    @Option(help: "Output path for the VoiceClonePrompt JSON")
    var output: String

    mutating func run() throws {
        let audioURL = URL(fileURLWithPath: refAudio)
        let waveform: [Float]
        if rawFloat32 {
            let data = try Data(contentsOf: audioURL)
            waveform = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        } else {
            waveform = try AudioInput.loadMono(url: audioURL, sampleRate: Double(MlxVoiceCloneEncoder.sampleRate))
        }
        let seconds = Double(waveform.count) / Double(MlxVoiceCloneEncoder.sampleRate)
        print(String(format: "Reference: %@ (%.2f s, %d samples)", audioURL.lastPathComponent, seconds, waveform.count))

        if !xVectorOnly, refText == nil {
            print("Warning: no --ref-text; ICL prefix assembly needs the reference transcript.")
        }

        let loadStart = Date()
        let encoder = try MlxVoiceCloneEncoder(
            modelDirectory: modelDir.map { URL(fileURLWithPath: $0) },
            maxReferenceSeconds: maxReferenceSeconds
        )
        print(String(format: "Encoders loaded in %.2f s", Date().timeIntervalSince(loadStart)))

        let encodeStart = Date()
        let prompt = try encoder.encode(waveform, includeReferenceCodes: !xVectorOnly, referenceText: refText)
        print(String(format: "Encoded in %.2f s", Date().timeIntervalSince(encodeStart)))

        let outputURL = URL(fileURLWithPath: output)
        let json = JSONEncoder()
        json.outputFormatting = [.sortedKeys]
        try json.encode(prompt).write(to: outputURL)

        print("x-vector dim: \(prompt.speakerEmbedding.count)")
        if let codes = prompt.referenceCodes {
            let quantizers = prompt.referenceCodeFrames > 0 ? codes.count / prompt.referenceCodeFrames : 0
            print("reference codes: (\(quantizers), \(prompt.referenceCodeFrames))")
        } else {
            print("reference codes: none (x-vector-only)")
        }
        print("Wrote \(outputURL.path)")
    }
}
