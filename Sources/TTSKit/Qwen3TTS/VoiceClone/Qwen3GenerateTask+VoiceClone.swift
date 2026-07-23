//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import Foundation

// MARK: - ICL voice-clone prefix assembly

extension Qwen3GenerateTask {
    /// Build the full ICL (in-context-learning) voice-clone prefix.
    ///
    /// Mirrors the Python reference `_build_icl_embeddings`: a dual-track
    /// sequence where each slot is the element-wise sum of a text-track and a
    /// codec-track embedding.
    ///
    /// Layout (`|` marks track alignment):
    /// ```
    /// text : [instr?] [role] [PAD×5] [BOS] | [refText + mainText + EOS] | [PAD × refFrames+1]
    /// codec: [zeros ] [zero] [think,thinkBos,lang,thinkEos,x-vector,PAD] | [PAD × textLen] | [BOS, refFrame₀…refFrameₙ]
    /// ```
    /// The trailing `codecBOS` of the 7-slot control block is dropped — the ICL
    /// block re-introduces it ahead of the reference RVQ frames. All synthesis
    /// text is baked into the prefix (no trailing text tokens during
    /// generation), and generation continues the codec track after the last
    /// reference frame.
    func buildVoiceCloneICLEmbeddings(
        prompt: VoiceClonePrompt,
        referenceText: String,
        lang: Qwen3Language,
        instruction: String?,
        textTokenIds: [Int32],
        embedDim: Int
    ) async throws -> (embeds: [[FloatType]], mainTextRange: Range<Int>) {
        guard let referenceCodes = prompt.referenceCodes, prompt.referenceCodeFrames > 0 else {
            throw TTSError.invalidConfiguration("ICL voice clone requires reference RVQ codes")
        }
        guard prompt.speakerEmbedding.count == embedDim else {
            throw TTSError.invalidConfiguration(
                "Speaker embedding dim \(prompt.speakerEmbedding.count) != decoder embed dim \(embedDim)"
            )
        }

        let zeroCodecEmbed = EmbedUtilities.zeroEmbed(dim: embedDim)

        // --- instruction (text track; codec zeros) ---
        var textTrack: [[FloatType]] = []
        var codecTrack: [[FloatType]] = []
        if let instruction, !instruction.isEmpty {
            let instructPrompt = "<|im_start|>user\n\(instruction)<|im_end|>\n"
            for tokenId in tokenizer.encode(text: instructPrompt).map({ Int32($0) }) {
                try await textTrack.append(textProjector.project(tokenId: tokenId))
                codecTrack.append(zeroCodecEmbed)
            }
        }

        // --- role prefix ---
        let rolePrefix = "<|im_start|>assistant\n"
        for tokenId in tokenizer.encode(text: rolePrefix).map({ Int32($0) }) {
            try await textTrack.append(textProjector.project(tokenId: tokenId))
            codecTrack.append(zeroCodecEmbed)
        }

        // --- control block: 7 codec slots minus the trailing BOS ---
        // [THINK, THINK_BOS, lang, THINK_EOS] via the code embedder, the raw
        // x-vector in the speaker slot, then PAD. The matching text track is
        // PAD × 5 + BOS (same as custom_voice: pads = 7 control slots - 2).
        let textPadEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)
        let textBosEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textBOS)

        var controlCodec: [[FloatType]] = []
        for codecId in [
            Qwen3TTSConstants.codecThink,
            Qwen3TTSConstants.codecThinkBos,
            lang.tokenID,
            Qwen3TTSConstants.codecThinkEos,
        ] {
            try await controlCodec.append(codeEmbedder.embed(tokenId: codecId))
        }
        controlCodec.append(prompt.speakerEmbedding.map { FloatType($0) })
        try await controlCodec.append(codeEmbedder.embed(tokenId: Qwen3TTSConstants.codecPAD))

        // Text side of the control block: pads for all but the last slot, then BOS.
        for _ in 0..<(controlCodec.count - 1) {
            textTrack.append(textPadEmbed)
        }
        textTrack.append(textBosEmbed)
        codecTrack.append(contentsOf: controlCodec)

        // --- ICL block ---
        // Text: reference transcript + synthesis text + EOS, over codec PAD.
        let codecPadEmbed = try await codeEmbedder.embed(tokenId: Qwen3TTSConstants.codecPAD)
        var iclTextEmbeds: [[FloatType]] = []
        let iclTokenIds = tokenizer.encode(text: referenceText).map { Int32($0) } + textTokenIds
        for tokenId in iclTokenIds {
            try await iclTextEmbeds.append(textProjector.project(tokenId: tokenId))
        }
        try await iclTextEmbeds.append(textProjector.project(tokenId: Qwen3TTSConstants.textEOS))

        // Absolute KV span of the SYNTHESIS text (main text) within the prefix —
        // the coverage region the guardrail anchor tracks. It sits after the
        // control block + the reference transcript, before the trailing EOS.
        let refTextCount = iclTokenIds.count - textTokenIds.count
        let mainTextStart = textTrack.count + refTextCount
        let mainTextEnd = mainTextStart + textTokenIds.count

        textTrack.append(contentsOf: iclTextEmbeds)
        codecTrack.append(contentsOf: Array(repeating: codecPadEmbed, count: iclTextEmbeds.count))

        // Codec: BOS + one summed embedding per reference RVQ frame, under text PAD.
        var iclCodecEmbeds: [[FloatType]] = []
        try await iclCodecEmbeds.append(codeEmbedder.embed(tokenId: Qwen3TTSConstants.codecBOS))
        let frames = prompt.referenceCodeFrames
        let quantizers = referenceCodes.count / frames
        for t in 0..<frames {
            var frame = [Int32](repeating: 0, count: quantizers)
            for q in 0..<quantizers { frame[q] = referenceCodes[q * frames + t] }
            try await iclCodecEmbeds.append(embedRVQFrame(frame))
        }

        codecTrack.append(contentsOf: iclCodecEmbeds)
        textTrack.append(contentsOf: Array(repeating: textPadEmbed, count: iclCodecEmbeds.count))

        assert(
            textTrack.count == codecTrack.count,
            "ICL track alignment mismatch: text=\(textTrack.count) codec=\(codecTrack.count)")

        // Prompt-fit validation against the CodeDecoder KV budget: the prefix
        // plus generated frames (~12/s of output audio) must fit.
        let prefixLength = textTrack.count
        let maxSeq = codeDecoder.kvCacheMaxSequenceLength
        if prefixLength >= maxSeq {
            throw TTSError.generationFailed(
                "ICL voice-clone prompt (\(prefixLength) tokens) does not fit the CodeDecoder KV cache "
                    + "(\(maxSeq)). Use a shorter reference clip or synthesis text."
            )
        }
        if maxSeq - prefixLength < 128 {
            Logging.error(
                "ICL voice-clone prompt uses \(prefixLength)/\(maxSeq) KV slots; only "
                    + "\(maxSeq - prefixLength) remain for generation (~12 frames per second of audio)."
            )
        }

        let embeds = zip(textTrack, codecTrack).map { EmbedUtilities.addEmbeddings($0.0, $0.1) }
        return (embeds, mainTextStart..<mainTextEnd)
    }

    /// Embedding of one 16-code RVQ frame: `codeEmbedder(code₀) + Σᵢ
    /// multiCodeEmbedder(codeᵢ + (i−1)·codecVocabSize)` — the same offsets the
    /// generation loop uses for its per-step codec hidden state.
    func embedRVQFrame(_ frame: [Int32]) async throws -> [FloatType] {
        var summed = try await codeEmbedder.embed(tokenId: frame[0]) as [FloatType]
        for i in 1..<frame.count {
            let offsetId = frame[i] + Int32(Qwen3TTSConstants.codecVocabSize * (i - 1))
            let embed = try await multiCodeEmbedder.embed(tokenId: offsetId) as [FloatType]
            summed = EmbedUtilities.addEmbeddings(summed, embed)
        }
        return summed
    }
}
