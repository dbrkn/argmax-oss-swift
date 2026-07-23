# Decoding Guardrails for the MLX Talker — production design (RD-655 port)

Ports rongxiang's RD-655 training-free decoding guardrails
(`argmaxinc/research-experiments@rongxiang/RD-655-…`) into TTSKit's MLX talker.
Long generations **skip** spans of the script or **hallucinate** (loop / re-read /
speak wrong content); both are detectable online from one attention signal and
fixable by rolling back and re-decoding with a corrective bias — no fine-tuning,
no model change. Upstream validated WER 11.7%→3.5% (0.6B long) and 21.5%→5.9%
(1.7B long) with do-no-harm on clean cells.

This document is the contract the implementation holds to. It is deliberately
explicit about production cost because this ships in `argmax-cli`/TTSKit, not a
notebook.

## Scope

- **In scope:** the MLX talker path (`TTSKitMLX`) — the CodeDecoder used by
  `--code-decoder-backend mlx`, which is the deployable hybrid and what the
  reflen/latency eval runs use. The talker attention is Swift/MLX we own, so the
  observable and the bias are implementable exactly as upstream.
- **Out of scope:** the CoreML talker (`Qwen3CodeDecoder`). Its attention is
  sealed inside the compiled `.mlmodelc`; per-head scores are not available at
  runtime. Guardrails there would require re-exporting the asset with attention
  outputs — a separate, larger effort. `GenerationOptions.guardrails` is a
  no-op (with a one-time warning) when the active talker is CoreML.

## The three layers (separable; ship in this order)

| layer | where | risk | cost |
|---|---|---|---|
| 1. observable `f(t)` | `TalkerAttention` (MLX) | audio-fidelity | per-step, small, measured |
| 2. coverage monitor | pure Swift, per-task | none (CPU logic) | negligible |
| 3. rollback + soft-align | generation orchestrator + `TalkerAttention` | correctness/latency | conditional, bounded |

Layers 1+2 are **observe-only** (detection + telemetry, cannot change audio).
Layer 3 is the fix and is the only part that alters output.

## The observable — `f(t)`

`f(t) = (argmax_text(anchor-head attention at step t) − prefill) / (T_text − prefill) ∈ [0,1]`
— normalized progress through the chunk's text tokens, from one emergent
alignment head (0.6B: L6H0; 1.7B: L3H0; **config, validated per checkpoint**).

Implementation: in `TalkerAttention.callAsFunction`, when the layer is the anchor
layer and it is a single-step decode, compute a **separate** softmax of
`q[anchorHead] @ k[anchorHead].T` over the text region `[prefill, T_text)` and
take its argmax. The real `MLXFast.scaledDotProductAttention` output is
**unchanged** — the observable is a side computation. Audio is bit-identical to
today (fidelity gate below).

### Production cost of the observable — the #1 latency question

- **Compute:** one extra `(1×d)·(d×T)` score + softmax for one head per step ≈
  0.1–0.3% of the full talker forward (28 layers × all heads). Immaterial in FLOPs.
- **The real cost is the host readback.** The monitor needs the argmax as an
  `Int` on the CPU each step to fire promptly. Reading a scalar off the GPU forces
  a sync that can break MLX's lazy pipelining. Mitigations, in the code:
  1. Read back **only one Int32 scalar/step** (the argmax bin), not the softmax.
  2. Fuse the argmax into the same `eval()` the decode step already forces when
     it reads the sampled token — the talker already syncs per step to sample, so
     the extra scalar rides that existing barrier (no *new* sync).
  3. A `guardrailsObserveStride` (default 1) can subsample if a machine shows
     overhead; the monitor tolerates it (bins are coarse).
- **Budget & gate:** observe-only overhead must be **< 3% wall-clock per step**,
  asserted by `GuardrailOverheadBenchmark`. If a target platform exceeds it, the
  stride knob or a batched-readback fallback keeps us in budget. **We publish the
  measured number; we do not ship an unmeasured "near-dense" claim.**

## The coverage monitor — Layer 2 (pure logic)

Direct port of `coverage_monitor.py`: bins the text (`TOK_PER_BIN=10`), tracks
dwell/bin against a P10 settle frontier, fires **skip** (coverage deficit) or
**hallucination** (coverage stall), each with a rollback step. Dimension-agnostic
(token units, not model units) so it ports byte-for-byte and carries the upstream
operating point unchanged. Bounded state: `O(n_bins + n_steps)` ≈ a few KB.
Parity-locked to upstream by `CoverageMonitorParityTests` (his exact defaults +
synthetic skip/stall/clean trajectories must fire identically).

## Rollback + soft-align — Layer 3

**Architectural note that shapes this layer.** Upstream owns the whole decode
loop in `gen_guarded.py`. In TTSKit the loop lives in the **generation
orchestrator** (`Qwen3GenerateTask`), and `MlxCodeDecoder.decode()` is a per-step
call. So rollback is implemented as a capability the orchestrator drives, not
inside the talker:

- **Checkpoint = trim, not snapshot.** The KV cache is append-only; "roll back to
  step n" trims each layer's cache and the internal MLX cache offset back to
  `prefill + n` — **O(1), no per-step copy** (a naive snapshot would be O(T²)
  memory — explicitly rejected). The orchestrator also truncates its generated
  code list and re-primes the monitor (`rollback_to`, replay = exact prefix state).
- **Intervention:** fresh (deterministic, seed-derived) reseed + a localized Huber
  attention bias on the anchor-layer heads over the rolled-back span
  `[resume, fire]`, off for the tail (`align.py` port). `λ=0.2, δ=10`.
- **Bounds (production safety — non-negotiable):**
  - `maxRetries` (default 10) hard cap on rollbacks per generation.
  - `maxRollbackSeconds` wall-clock budget for guardrail work; exceed → **give up
    and return the un-guarded generation** (degrade to baseline, never hang).
  - No "chase-escalation" (upstream shelved it: one case went 8→600 deletions).
    A future gated rescue must be revert-if-worse, not a blanket deepening.
- **Determinism:** each retry's reseed is a documented function of the base seed +
  retry index, so runs remain reproducible.

## Fidelity (must-hold invariants)

1. **Guardrails OFF ⇒ bit-identical to today.** `GuardrailFidelityTests`: same
   text/seed, guardrails off vs the flag absent → identical codes. The observable
   never touches the sampler RNG or the SDPA output.
2. **Observe-only (Layers 1+2, no rollback) ⇒ also bit-identical.** Detection and
   telemetry cannot change audio. This is what makes Stage 1 zero-risk to ship.
3. **λ=0 / disarmed bias ⇒ bit-identical.** The bias tensor is exactly zero when
   not armed.

## Concurrency

The MLX talker already forces `--concurrent-worker-count 1` (single private KV
cache). Guardrail state (monitor, soft-align integrator, observable buffers) is
**per-generation**, created by the orchestrator per task and threaded through the
decode call — never mutable state on the shared talker module. No locks.

## Configuration (`GenerationOptions.guardrails`, off by default)

`enabled` · `anchorLayer`/`anchorHead` (checkpoint-specific; validated at load,
loud error if out of range) · `biasHeads` · detector operating point
(`tokPerBin, commitFrac, win, settlePctl, fairWin, fairInit, overFrac, stallFrac`)
· `lambda, delta, stride` · `maxRetries, maxRollbackSeconds` · `observeStride`.
Ships with upstream-validated defaults; **off** until we validate on our reflen set.

## Telemetry (the run metrics the study needs, and prod monitoring)

On the generation result (alongside `SpeechTimings`):
`fired: Bool` · `rollbacks: Int` · `failures: [(type, fireStep, rollbackStep)]` ·
`rollbackAudioSeconds: Double` (added work) · `hitRetryCap: Bool` ·
`fTrajectory: [Float]?` (opt-in, for offline analysis / anchor validation).
Surfaced through the CLI and the OpenBench sink so a run reports **fire rate,
per-type breakdown, times-fired, WER/wSIM before-vs-after, and do-no-harm
regressions** without extra plumbing.

## Anchor-head validation

Defaults (L6H0 / L3H0) were calibrated on the mlx-community bf16 checkpoints we
generate from, so they should transfer. Stage 1 confirms empirically: the
`fTrajectory` on a known-good generation must be monotone and near-full text
coverage. If a checkpoint/quant moves the head, `analysis/anchor_sweep` (port of
`vc_icl_allhead_rec.py`) re-finds it once; it's a config change, not a code change.

## Interaction with chunking

Production chunks text (~42-token chunks); `f(t)` and `T_text` are **per chunk**.
Short chunks make catastrophic long-run skips rarer than upstream's single-shot
long generations — so our fire rate may be lower and the guardrail's marginal
value smaller than the headline numbers. **Stage 1 measures this honestly**
before we invest in Stage 2 on our workload.

## Rollout

1. **Stage 1** (observe-only): observable + monitor + telemetry, off by default.
   Ship, run reflen-100 with `guardrails.enabled + observeOnly`, report fire rate
   and correlation with the existing WER/wSIM outliers. Zero audio risk.
2. **Stage 2** (fix): rollback + soft-align, bounded. Re-run, report the
   before/after WER/wSIM, fire counts, and regressions. Flip the default only if
   do-no-harm holds on our set.
