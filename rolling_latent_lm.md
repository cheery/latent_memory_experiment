# Handoff: rolling latent compression LM

You are picking up `rolling_latent_lm.py`, a single-file PyTorch research prototype.

> **Status (2026-11-16):** it runs, the selftest passes on CPU, XPU and CUDA, and it has been trained on enwik8 on Modal. The original design collapsed (latents ignored); see **§9 Results so far** before reading the plan below. Work has moved on to `rolling_latent_qwen.py`, the same method on a frozen pretrained Qwen3 base. Sections 1–8 are the original handoff, kept for the design rationale; defaults have changed where §9 says so.

## 1. The idea in one paragraph

Keep only a short raw window of text in the model's context. Continuously compress older text, block by block, into a few latent vectors. Train the compression so the model *behaves the same* with the compressed context as it does with the full raw context (self-distillation with KL), rather than training it to reconstruct text. Because lossy compression can never guarantee identical behavior for every future query, the raw text is kept in storage outside the context. A learned head estimates when the latents are insufficient, and a learned pointer picks which block to re-expand to raw text ("recall").

Related work it draws on: Coca-Cola (arXiv 2611.06500, action-preserving training of latent tool-output compression), Memorilla (arXiv 2610.21375, query-conditioned latent memory), AutoCompressors / Activation Beacon (the same model writes its own summary vectors).

## 2. Architecture

One causal byte-level transformer with RoPE plays every role.

| Role | Input sequence | Output |
|---|---|---|
| Teacher | full raw `[old blocks | recent window | future]` | next-byte logits |
| Compressor | `[latents of earlier blocks | block i raw | conditioning window | K beacon tokens]` | beacon hidden states -> MLP adapter -> K latents |
| Student | `[latents (or one block raw = "expanded") | recent window | future]` | next-byte logits, divergence head, pointer head |

Details worth knowing before debugging:

- **Conditioning window.** Normally the W bytes immediately after the block, which is what a streaming model sees at eviction time. In "recond" batches it is the final recent window, which is what re-compression at inference uses.
- **Latent positions.** The K latents of a block get RoPE positions at the ends of the block's K sub-spans, so relative distances to later text match the raw layout. RoPE is relative, so inference shifts all positions down to keep numbers small.
- **No leakage, by construction.** Latents may encode the recent window, because it is their conditioning. So student losses are computed **only on the future region**, never on positions inside the recent window. The selftest checks this. Do not "fix" it by scoring the recent window.
- **`future == block` by default.** This is on purpose. At inference the recent window varies between W and W+B-1 bytes before eviction, and training tail length W+F matches that upper end.

## 3. Losses

| Name | What | Default weight |
|---|---|---|
| `lm` | teacher next-byte CE on the whole raw sequence (keeps the base LM trained) | 1 |
| `ce_s` | student CE on the future region | `w_ce=1.0` |
| `kl` | KL(teacher ‖ student) on the future region, teacher detached, ramped in | `w_kl=1.0` |
| `div` | MSE of div head vs `log1p(KL)` per position | `w_div=0.1` |
| `ptr` | on copy probes, CE of pointer over blocks vs the block holding the byte being predicted | `w_ptr=0.5` |

The student only trains after `distill_start` steps (default 1000) of plain LM training, so the teacher is worth imitating.

**Copy probes** (`probe_frac` of batches): the future region becomes `[PROBE] + verbatim span copied from the old region`. The teacher can copy from raw context. The student must keep the detail in latents or recall the block. This is the stress test for "does compression lose exact detail."

## 4. Steps

### 4.1 Selftest (CPU is fine, about a minute)

```bash
pip install torch numpy
python rolling_latent_lm.py selftest
```

It checks: teacher causality; student causality over the future region; that old-block content reaches student predictions through latents; expanded-layout shapes; finite losses in all 8 batch modes (probe × expanded × recond); gradients reaching beacons, adapter, div head and pointer; overfitting a fixed batch (loss must fall below 70% of start); `evaluate` runs; streaming generation with forced recall and re-compression.

**Fix whatever fails, and keep the selftest passing.** If you change a check, explain why in your report.

### 4.2 Smoke run (GPU, minutes)

Any text file of a few MB works, for example tiny Shakespeare or a slice of enwik8.

```bash
python rolling_latent_lm.py train --data small.txt --out smoke.pt \
  --steps 2000 --distill_start 300 --kl_ramp 300 --eval_every 500 \
  --d_model 256 --n_layers 4 --n_heads 4
```

Expect `lm` to fall steadily, and `kl` and `ce_s` to appear after step 300 and fall.

### 4.3 Real run

Use enwik8 (100 MB, standard for byte-level LMs).

```bash
python rolling_latent_lm.py train --data enwik8 --out run1.pt --steps 30000
python rolling_latent_lm.py eval --ckpt run1.pt --data enwik8
python rolling_latent_lm.py generate --ckpt run1.pt --prompt_file some_long_text.txt --n 2000 --tau 0.5
```

Defaults: d=512, 8 layers (~25M params), blocks of 256 bytes → 16 latents (16× compression), 6 old blocks, 256-byte window, 256-byte future. Teacher context 2048 bytes, student context 608 (864 with one expanded block). Each step costs one teacher pass, six sequential compressor passes and one student pass. Expect roughly 2–3× the cost of a plain LM step.

## 5. What to measure (`eval` prints these)

| Metric | Meaning | What good looks like |
|---|---|---|
| `teacher_bpb`, `student_bpb` | bits per byte on the future region | Student close to teacher. The gap is the cost of compression. |
| `kl_nats` | mean KL(teacher ‖ student) | Falls during training. This is the core "behavior preserved" number. |
| `div_corr` | correlation of predicted vs actual log1p(KL) | Clearly positive (>0.3 would be encouraging). Near 0 means the model does not know what it lost, and recall cannot work. |
| `copy_acc_teacher` | exact-copy accuracy with raw context | Should get high. If not, the base LM is too weak or too short-trained to evaluate anything else. |
| `copy_acc_none` | student, no recall | Expected to be clearly below teacher. This measures detail loss. |
| `copy_acc_oracle` | student with the correct block expanded | Should approach teacher. If not, the student cannot read mixed raw/latent layouts. |
| `copy_acc_pointer` | student with the pointer-chosen block expanded | Between none and oracle. The gap to oracle is pointer error. |
| `pointer_hit` | pointer picks the oracle block | Chance is 1/6. |
| `div_pred_probe` vs `div_pred_natural` | predicted divergence on probes vs normal text | Probe should be higher. That is the signal that triggers recall. |

**Baseline you should add:** train the same config with `--w_kl 0` (CE only, no distillation) and compare `kl_nats`, `student_bpb` and copy accuracy. Without this you cannot tell whether behavior-preserving training does anything beyond plain student CE.

## 6. Known risks and likely failure points

- **Untested code.** Most likely problem areas: dtype mismatches under bf16 autocast (latents vs embeddings are cast to the embedding dtype in `compress` and `student`), `torch.load` behavior across versions, and shape errors in `pointer_logits` with boolean masking.
- **Teacher and student share weights.** The KL target moves as the model trains. If training is unstable after `distill_start`, try a lower `w_kl`, a longer `kl_ramp`, or add an EMA teacher (not implemented).
- **Student gradients flow into the shared trunk** and may degrade the teacher. Watch whether `lm` gets worse after `distill_start`. If it does, try `--detach_prev` or a lower `w_ce`/`w_kl`.
- **Compression collapse.** If `copy_acc_none` stays at the level of the closed-book baseline and the student ignores latents, check that latent norms are comparable to token embedding norms. The adapter output has no explicit scaling.
- **Pointer trained only on probes.** Natural-text recall at inference relies on the pointer generalizing from probes. That is an assumption, not something the code guarantees.
- **Context eviction.** At inference only the last `n_blocks` blocks are visible. Older blocks stay in the raw store but are unreachable. Recall from outside the visible context is not implemented.
- **Inference has no KV cache** and recomputes the full context for every generated byte. Fine for evaluation, slow for long generation.

## 7. Ablations worth running once the base run works

1. `--w_kl 0` (the baseline in section 5). Most important.
2. `--k_lat 8`, `16`, `32` (compression 32×, 16×, 8×). Plot `kl_nats` and `copy_acc_none` against compression.
3. `--expand_prob 0` vs default. Does training on mixed layouts matter for `copy_acc_oracle`?
4. `--recond_prob 0` vs default, then evaluate with `generate --recompress_every`.
5. `--probe_frac 0`. Does detail retention on natural text suffer without probes?

## 8. What to report back

- Selftest output and any bugs you fixed, with a one-line reason each.
- Loss curves (`lm`, `ce_s`, `kl`, `div`, `ptr`) and the final `eval` table for the main run and the `--w_kl 0` baseline.
- Steps per second and GPU memory at the default config.
- A generation sample with `--tau` set, including the printed recall events.
- Your judgment, plainly: does the compressed student track the teacher, does the divergence head know what it lost, and does recall recover the lost detail?

## 9. Results so far (2026-11-16)

### Bugs fixed

- **XPU device lost** (Intel Arc, torch 2.15.0+xpu): PyTorch's `_foreach_*` kernels fault this GPU. Default (foreach) AdamW and `clip_grad_norm_` both used them. Fixed with `fused=True` and `clip_grad_norm_(..., foreach=False)`. Not a bug in the model code. Repeated faults froze the laptop, so debug such crashes on CPU (`RLLM_DEVICE=cpu`).
- **CUDA support** for Modal (`modal_rllm.py`), peak-memory logging.
- No selftest check was weakened. Added checks: window-dropout batch modes, KV-cached generation matches uncached.

### Main run and `--w_kl 0` baseline (default config, 26M, enwik8, stopped at ~17k of 30k steps)

The two runs were indistinguishable at every eval. Larger eval of the main run (20 batches, 128 probes):

| metric | value |
|---|---|
| teacher / student / **no-memory** bpb on the future region | 1.241 / 1.286 / **1.287** |
| copy acc teacher / student (no recall) / **no memory** | 0.960 / 0.724 / **0.722** |
| copy acc oracle block expanded / pointer's block expanded | 0.903 / 0.783 |
| pointer picks a block the span touches (`pointer_hit_any`) | 0.48 |
| div head: predicted log1p KL on probes / natural text | 0.36 / 0.018 |

`diag` on the step-17k checkpoints:

| | main | `--w_kl 0` |
|---|---|---|
| latent RMS / token-embedding RMS | **630x** | **426x** |
| future bpb, latents + window | 1.291 | 1.302 |
| future bpb, **latents only, no window** | **1.437** | **1.489** |
| future bpb, nothing (future only) | 1.436 | 1.471 |

**Compression collapse.** The latents carried no usable information. They grew to hundreds of times the embedding scale (the adapter output was unnormalised) and attracted attention anyway, probably as attention sinks. "Student close to teacher" came entirely from the recent window. Since the latents did nothing, the KL-vs-CE comparison says nothing about distillation.

### Fixes and a 5k-step rerun

- `lat_norm` (default on): latents RMS-normalised to the current token-embedding RMS. Old checkpoints load without it.
- `--drop_window_prob 0.25`: the student sees only `[latents | future]` in a quarter of batches, so the future can only be predicted through the latents.
- `--probe_frac 0.5` (was 0.25).
- New eval columns: `no_memory_bpb`, `no_window_bpb`, `no_window_no_memory_bpb`, `copy_acc_no_memory`, `pointer_hit_trained`, `pointer_hit_any`.

| step | no-window bpb with / without latents | with-window bpb with / without latents | copy acc student / no memory |
|---|---|---|---|
| 1000 | 2.021 / 2.045 | 1.893 / 1.895 | 0.60 / 0.60 |
| 3000 | 1.736 / 1.702 | 1.538 / 1.536 | 0.70 / 0.70 |
| 5000 | 1.590 / 1.603 | 1.436 / 1.438 | 0.71 / 0.70 |

The latents now carry a little (about 0.01–0.02 bpb when they are the only context), but the student with a window gains nothing from them, and exact detail is not retained.

### Judgment on the questions in §8

- **Does the compressed student track the teacher?** Only trivially: it tracks it as well as a model with no memory does. Older context is worth just 0.046 bpb on enwik8 at this scale, so there is little to preserve.
- **Does the divergence head know what it lost?** Partly: `div_corr` reaches 0.27–0.32, and predicted divergence on probes is 10–20x that on natural text.
- **Does recall recover lost detail?** Yes, when the right block is chosen (copy 0.72 -> 0.90, teacher 0.96). The pointer is the weak part.
- **Is KL distillation better than CE alone?** Not measurable here, since the latents were unused.

### Other changes

- `generate` has a KV cache, rebuilt only on eviction, recall and re-compression; output identical to the uncached path, ~12x faster on CPU.
- `diag` command: latent vs embedding norms, attention mass on memory, latents-only prediction.
- Costs on an L40S (default config): 120 ms per teacher-only step, 296 ms per student step, 20.4 GB.

### Next: `rolling_latent_qwen.py`

Same method on a frozen Qwen3-0.6B-Base (teacher) with LoRA + compressor (student), with a passkey eval. After 200 steps: teacher copy 0.98 and passkey 0.88 (real long-context behaviour to preserve); passkey 0.94 with the right block expanded; pointer picks a span block 75% of the time; latents alone don't hold the key yet. Pitfall: transformers 5.x reads gaps in `position_ids` as packed-sequence boundaries unless an explicit `attention_mask` is passed. After 3000 steps: memory gain 0.015 nats with the window, 0.033 without; KL 0.143; copy acc student 0.584 vs no memory 0.582; passkey from latents 0, with the right block expanded 1.00; pointer picks a span block 66% of the time; div_corr 0.34. The latents carry gist, not detail; recall carries the detail, and the pointer is the bottleneck. See `RESULTS.md` for the table.

