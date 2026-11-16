# Experiment log

The full record of the rolling latent memory experiments, in the order they were run, including
the dead ends. `latent_memory.md` distils the working design; this file keeps the evidence.
All runs on Modal (L40S / H100) unless noted. "nats" and "bpb" are per byte for the byte-level
models; EM / F1 are HotpotQA exact match and token F1.

## Rolling latent compression (rolling_latent_lm.py, rolling_latent_qwen.py)

Keep a short raw window in context and compress older text, block by block,
into a few latent vectors written by the same model. Train the compressed
student to behave like the model reading the full raw context (KL), keep the
raw text outside the context, and let a divergence head and a pointer decide
when to re-expand ("recall") a block. Design and handoff notes in
`rolling_latent_lm.md`; all runs on enwik8 on a Modal L40S.

**Byte level, 26M parameters, from scratch.** 6 blocks of 256 bytes, 16
latents each (16x), 256-byte window, 256-byte scored future.

| Run (step) | teacher bpb | student bpb | no-memory bpb | copy acc teacher / student / no memory |
|---|---|---|---|---|
| main, KL + CE (~17k) | 1.241 | 1.286 | 1.287 | 0.96 / 0.72 / 0.72 |
| fixed latents (5k) | 1.380 | 1.436 | 1.438 | 0.95 / 0.71 / 0.70 |

- **The first runs collapsed.** The student looked close to the teacher, but
  only because of its recent window: removing the latents changed nothing.
  The latents had grown to 400–600x the token-embedding RMS and drew
  attention without carrying content (attention sinks). With the window
  removed, predicting from the latents was no better than from nothing
  (1.437 vs 1.436 bpb). The `--w_kl 0` baseline was identical, so KL
  distillation did nothing beyond student CE.
- **Fixes:** latents RMS-normalised to embedding scale, window dropout (25%
  of batches give the student only latents + future), more copy probes.
  Afterwards the latents carry a little (no window: 0.013–0.023 bpb better
  than nothing), but with the window the gain is 0.002 bpb and copy accuracy
  is unchanged.
- **The test bed is the problem as much as the method:** on enwik8, all 1536
  bytes of older context are worth only 0.046 bpb to the teacher, so there
  is little behaviour to preserve.
- **Recall works:** re-expanding the right block takes copy accuracy from
  0.72 to 0.90 (teacher 0.96). The pointer is weak (picks a block the span
  touches 19–48% of the time). The divergence head does react to probes
  (predicted log1p KL 0.36–0.39 vs 0.02–0.04 on natural text).
- **Speed:** a student step costs about 4.5x a plain LM step (six
  sequential compressor passes). Generation now has a KV cache rebuilt only
  on eviction, recall and re-compression: identical output, 12x faster.

**Frozen Qwen3-0.6B-Base, LoRA r=16 + compressor (12.5M trainable).** 4
blocks of 256 tokens, 16 latents each, 256-token window and future. 670
ms/step at batch 4, 36.5 GB. After 200 steps (small eval):

- **The teacher now has real long-context behaviour:** copy accuracy 0.98,
  passkey (a 5-digit key in an old block, asked for at the end of the
  window) 0.88.
- **Recall works well:** passkey 0.94 with the right block expanded, and the
  pointer picks a block the copied span touches 75% of the time.
- **Latents alone don't hold detail:** passkey 0, copy accuracy equal to
  no memory.

After 3000 steps (35 min, about $1.50; 10 batches, 32 probes, 32 passkeys):

| | value |
|---|---|
| CE nats: teacher / student / no memory | 1.946 / 1.901 / 1.916 |
| no-window CE nats: with / without latents | 2.116 / 2.149 |
| KL to teacher | 0.143 |
| copy acc: teacher / student / no memory | 0.989 / 0.584 / 0.582 |
| copy acc: right block expanded / pointer's block | 0.892 / 0.744 |
| passkey: teacher / latents only / right block expanded | 0.91 / 0 / 1.00 |
| pointer picks a block the span touches | 0.66 |
| div head: correlation, predicted on probes / natural | 0.34, 0.27 / 0.12 |

- **The latents carry gist, not detail:** a steady 0.015 nats with the
  window and 0.03 without, already about 10x the byte model's gain, but no
  copied span or passkey survives compression.
- **Recall carries the detail:** the right block expanded gives every
  passkey. The pointer is the bottleneck.
- **The div head ranks well but its level drifts** (mean prediction on
  natural text went 0.02 -> 0.43 -> 0.12 between evals), so a fixed recall
  threshold is unreliable; set tau from validation quantiles.
- **The copy probe asks a one-pass compressor for too much:** 16 latents
  *can* hold a 256-token span exactly (see "Cramming text into latents"
  below), but only optimised per block; a compressor trained for 3000 steps
  learned to keep gist and gave up on detail.
- **Pitfall:** transformers 5.x treats a gap in `position_ids` as the start
  of a new packed sequence when no `attention_mask` is passed, and blocks
  attention across it. Latents sit at gapped positions, so nothing could see
  them. Always pass an explicit all-ones mask.

**Cramming text into latents (`cram.py`).** How much text can the frozen,
unmodified Qwen3-0.6B-Base regenerate exactly from a few input vectors?
Phase A replicates Kuratov et al. (2025, "Cramming 1568 tokens into a single
vector"): for each held-out enwik8 span, m vectors are optimised by gradient
descent (Adam, lr 0.01, up to 3000 steps) so that the model reading only
`[vectors]` reproduces the span under teacher forcing. 100% argmax accuracy
means greedy decoding from the vectors regenerates the span exactly. 8 spans
per cell, L40S.

| m | N tokens | ratio | acc | exact | tokens before 1st error | bits carried / needed | steps |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 128x | 0.94 | 0.25 | 44 | 385 / 454 | 3000 (cap) |
| 1 | 256 | 256x | 0.78 | 0.12 | 35 | 505 / 901 | 3000 (cap) |
| 1 | 512 | 512x | 0.69 | 0 | 7 | 619 / 1715 | 3000 (cap) |
| 1 | 1024 | 1024x | 0.64 | 0 | 45 | 508 / 3004 | 3000 (cap) |
| 16 | 256 | 16x | 1.00 | 1.00 | 256 | 896 / 901 | 281 |
| 4 | 128 | 32x | 1.00 | 1.00 | 128 | 450 / 454 | 625 |
| 4 | 256 | 64x | 1.00 | 1.00 | 256 | 895 / 901 | 1302 |
| 4 | 512 | 128x | 0.99 | 0.62 | 390 | 1650 / 1715 | 3000 (cap) |

- **One vector of this 0.6B model carries about 500–600 bits** (a few dozen
  exactly recoverable tokens); the 1024-token row is under-optimised, not a
  smaller capacity. Several vectors scale it: **4 vectors hold 256 tokens
  exactly (64x)**, and at 128x (512 tokens) 5 of 8 spans still come back
  exactly within 3000 steps (~98 tokens per vector).
- **Optimised vectors sit at ~3.3x the token-embedding RMS.**

Phase B trains an encoder instead: the base model + LoRA (r=16, 12.2M
trainable) reads `[span | k beacons]` once; an adapter maps the beacon
states to k latents (RMS-normalised, learned scale); the frozen model with
LoRA off decodes, in the same format and on the same held-out spans as
Phase A. 2000 steps, batch 8, lr 2e-4. Then Phase A's optimisation is
started from the encoder's latents ("refine").

| | encoder alone | + 50 steps | exact after | Phase A from random |
|---|---|---|---|---|
| k=16, 256 tokens (16x) | acc 0.57, 133 bits | acc 0.92, 744 bits | 110–159 steps | 281 steps |
| k=4, 128 tokens (32x) | acc 0.56, 57 bits | acc 0.94, 394 bits | 93–99 steps | 625 steps |

- **A one-pass encoder is far from the ceiling** (6–15% of the information
  after 2000 steps, still rising). Its latent scale barely moved from 1.0
  (1.00 -> 1.10) while the optimum is ~3.3x, which capped it; `--init_scale`
  and separate learning rates now exist for a second try.
- **As a starting point it works:** encoder + about 100 optimisation steps
  per block gives exact recall, 2–6x faster than from random vectors. So
  "store a block as a few latents, regenerate the exact text on recall" is
  feasible today, at ~100 forward/backward passes per block (fine offline or
  asynchronously at eviction).
- **Untested:** whether latents optimised to make the model *recite* a block
  also work as soft memory among other context (gist, relevance).

**Recall decisions (`rolling_latent_qwen.py recall_eval`).** Per position
of the scored region (16 natural, 16 copy-probe and 16 passkey examples,
12,288 positions), ground truth from actually expanding each block; recall
"helps" where a block gains > 1 nat (19% of positions: probes 47%, passkey
answers 6%, natural text 4%). First run on `q06_3k` (reader with LoRA):

| when (AUROC, "recall helps") | all | natural | passkey | probes |
|---|---|---|---|---|
| student entropy | **0.77** | 0.72 | 0.73 | **0.93** |
| TokUR total uncertainty (σ 0.1, M 2) | 0.75 | 0.71 | 0.70 | 0.91 |
| TokUR epistemic uncertainty | 0.68 | 0.71 | 0.55 | 0.78 |
| trained div head | 0.76 | 0.74 | 0.75 | 0.82 |

- **When to recall is mostly solved by plain entropy.** TokUR's epistemic
  uncertainty (Zhang et al., ICLR 2026) is the weakest signal at the
  paper's defaults; a sigma sweep is pending.
- **Frozen reader, first attempt (`q06_frozen`) regressed:** latents made
  predictions worse than no memory (step 1500: 2.286 vs 2.160 nats with the
  window, 2.607 vs 2.446 without). With the reader frozen, the pointer and
  div-head losses could only improve by reshaping the latents, at the LM's
  expense (in `q06_3k` the reader's LoRA absorbed that). Fix: the recall
  heads read detached hidden states (`head_grad` off by default). The
  broken run did recover by step 3000 (+0.002 / +0.020 nats), so the dip
  was mid-training, but it ended with nearly useless latents.
- **`q06_frozen2` (frozen reader, detached heads) is the best context
  latent so far:** at step 3000, CE 1.974 with latents vs 2.027 without
  (+0.054 nats with the window, +0.129 without; `q06_3k`: +0.015 /
  +0.033), KL 0.092 (lowest), copy accuracy 0.576 vs 0.567. Passkey from
  latents is still 0 (0.91–0.97 with its block expanded).
- **Which block is the bottleneck:** at helpful positions the trained
  pointer picks the best block 28% of the time, attention mass 18% (chance
  25%; attention sinks pull it to block 0). Recalling at the top 10% of
  positions by entropy captures 25% of the available gain with the oracle
  block, 8% with the trained pointer, 5% with attention.

**Second recall-decision run, all pointers and a TokUR sweep.** On
`q06_3k`: the label-free **lookahead pointer** (expand each block, keep
the one leaving the lowest entropy) picks the best block 93% of the time
(trained 28%, attention 18%, sink-corrected attention 20%, chance 25%),
and with an entropy trigger captures 88% of the oracle's gain (top 10%
of positions: 0.217 vs 0.246). TokUR gets worse as sigma grows (AUROC
0.768 at 0.05 to 0.533 at 0.5); its total uncertainty converges to plain
entropy (0.773) and its epistemic part is never better: **plain entropy is
the trigger to use here.**

**No-latents control** (`--latents none` on `q06_frozen2`, same examples:
window only, recall inserts the raw block). Mean CE after entropy-triggered
lookahead recall:

| recall budget | 0 | 5% | 10% | 20% |
|---|---|---|---|---|
| latents | 2.150 | 2.046 | 1.968 | 1.841 |
| no latents | 2.208 | 2.096 | 2.023 | 1.880 |

The recall machinery does not depend on the latents (entropy AUROC 0.769
vs 0.768, lookahead 0.91 vs 0.90); the latents add a steady 0.04–0.06
nats of gist on top.

**Archive latents are not memory** (`--latents cram`): Phase-A latents
that regenerate their block exactly (100%) roughly double the reader's CE
when placed in context (natural 4.00 vs 2.01 with no memory) and even
spoil recall (passkey with its block expanded 0.44 vs 1.00). They act as
"recite this" triggers that hijack what follows. "Contains the
information" and "useful as memory" are different properties. As storage
they are also pointless: 4 bf16 latents for 256 tokens take 8 KB, about
8x the raw text, and decoding is slower than re-reading the text.

**Demo** (`demo`, `q06_frozen2`, a 51-token English text, prompt "The
above repeated:"): raw text is repeated exactly; the compressor's latents
yield Wikipedia markup ("* [[http://en.wikipedia.org/wiki/1960s]]…"),
and asked for Alice's code they produce "The [[Secret Code of Alice]] is
a …": the topic survives, the content does not (and the compressor never
saw plain English or short blocks). Cram latents make the model say "The
above text is a bit unclear and fragmented".

**End to end: needles in long documents** (`needle_eval`, `q06_frozen2`):
4 "secret code of NAME" facts in random blocks of enwik8 filler, asked
after the window, exact match of the 5-digit code, 64 questions.

| mode | 16 blocks (4.4k tok) | 64 blocks (16.7k tok) |
|---|---|---|
| full raw document, base model | 0.80 | 0.80 |
| latents + the right block (oracle) | **1.00** | **1.00** |
| latents + entropy + lookahead over all blocks | 0.61 | 0.48 |
| no latents + entropy + lookahead | 0.33 | 0.34 |
| latents + trained pointer top-4 + lookahead | 0.27 | 0.09 |
| window only / latents without recall | 0.00 | 0.00 |
| reader tokens per recall (full lookahead) | 12.9k | 99.9k |

- **A short context with the right block beats reading everything** (1.00
  vs 0.80), as the "you don't need long context" paper argues.
- **Latents help the lookahead** (0.61 vs 0.33, 0.48 vs 0.34) even at 4x
  the distances seen in training.
- **Candidate retrieval is the unsolved part:** choosing among 64 blocks
  succeeds less than half the time and the full lookahead costs 6x reading
  the whole document; the trained pointer is at chance. Next: a cheap
  retrieval step (dense, and lexical as a baseline) feeding a 4-way
  lookahead, on questions without literal overlap.

**Instruct model, question answering (`rolling_latent_chat.py`, Qwen3.5-9B).**
Qwen3.5-9B is a hybrid model (24 Gated DeltaNet linear-attention layers, 8
full-attention layers, partial M-RoPE, 248k vocabulary); the frozen model
reads, a LoRA'd copy (74.8M trainable) compresses. Samples are chat turns:
HotpotQA (distractor) paragraphs, padded with other questions'
paragraphs, as 4 blocks of 256 tokens (16 latents each, 16x) + a 128-token
raw window, then the question; loss on the answer tokens only (CE + KL to
the model reading the raw documents), thinking off. Half the samples
compress the blocks knowing the question, a quarter drop the window.
H100 with flash-linear-attention: ~4–6 s per step of 4 questions, 37 GB.

| 32 val questions, EM (F1) | step 250 | 500 | 750 | 1000 |
|---|---|---|---|---|
| full documents (~1.4k tokens) | 0.44 (0.57) | 0.44 | 0.44 | 0.44 |
| window only | 0.09 (0.11) | 0.09 | 0.09 | 0.09 |
| **latents** | **0.31 (0.34)** | 0.28 (0.34) | 0.25 (0.31) | 0.25 (0.31) |
| latents compressed knowing the question | 0.31 (0.35) | 0.28 (0.34) | 0.25 (0.33) | 0.25 (0.31) |
| latents + gold blocks expanded | 0.59 (0.72) | 0.56 | 0.56 | 0.59 (0.72) |

- **First time latents carry answer-level content:** at step 250 they
  recover about two thirds of what the full documents add over the window
  (±0.08 noise at 32 questions); it drifted to about half by step 1000
  (10 -> 8 of 32 correct), so the best checkpoint is early. With Qwen3-0.6B
  the passkey from latents was always 0. Question-conditioned compression
  is no better than streaming compression here.
- **Latents + the gold blocks beat reading everything** (0.59 vs 0.44 EM).
- Not attributable to size alone: the objective (answer tokens only, not
  enwik8 continuation), the task, the reader (4096-d latents, instruct
  model) and the data all changed at once. A same-objective run on a
  small model would separate them.

**What the 9B actually answers** (`chat_demo`, step-1000 checkpoint, 5
questions, 8-block documents of ~2k tokens; recall = first-token entropy,
dense top-4, lookahead, up to two blocks):

| question (gold) | full | window only | latents | recall (blocks) |
|---|---|---|---|---|
| Speed of Sound album cover (Baudot code) | ✓ | "documents do not contain…" | "a photograph of a man's face" | ✗ [0, 7], evidence [0, 5, 6] |
| composer born 1750 (Antonio Salieri) | ✓ | Joseph Haydn | Joseph Haydn | ✓ [6] |
| April Margera's channel owner (Viacom Media Networks) | ✓ | Time Warner | "ViacomCBS" | ✓ [0, 2] |
| published in France (Avantages) | ✓ | "Neither…" | ✓ | ✓ |
| Full Frontal star (Eric Banadinović) | ✓ "Eric Bana" | Stephen Merchant | Stephen Merchant | ✗ [0], evidence [2, 4, 5] |

Gold blocks expanded: 5/5. Thinking on: full 5/5, latents 0/5, recall 3/5
(the same three).
- **Latents give gist** ("ViacomCBS" where the window alone says "Time
  Warner"; Avantages answered from latents alone), and fail cleanly (the
  same wrong guess as no memory for Salieri).
- **Recall turns gist into exact answers when retrieval finds the block;**
  both misses are retrieval misses.
- **With thinking on, the model reasons the latents away.** Its traces
  say things like "a set of documents (though only one snippet is
  provided in the prompt…)": deliberate reasoning only trusts readable
  text, so latents that help direct answers are ignored when thinking (the
  compressor was trained on direct answers only). Recall works in both
  modes.
**`chat_eval`, 8-block documents (~2k tokens), step-1000 checkpoint, 32
questions,** EM (F1); recall = first-token entropy -> top-4 by BM25 or
dense -> lookahead, up to two blocks; thinking budget 256 tokens:

| mode | thinking off | thinking on |
|---|---|---|
| full documents | 0.41 (0.59) | 0.44 (0.60) |
| window only | 0.09 (0.13) | 0.06 (0.10) |
| latents | 0.28 (0.34) | 0.16 (0.22) |
| latents, question-conditioned | 0.28 (0.34) | 0.22 (0.25) |
| latents + gold blocks (oracle) | 0.63 (0.76) | 0.50 (0.66) |
| **latents + recall, BM25** | **0.50 (0.64)** | 0.34 (0.51) |
| latents + recall, dense | 0.44 (0.54) | 0.31 (0.46) |

| retrieval | BM25 | dense (mean-pooled 9B states) |
|---|---|---|
| top-4 contains some gold block | 1.00 | 0.94 |
| top-4 contains all gold blocks | 0.63 | 0.31 |
| recall picked a gold block | 0.81 | 0.63 |
| blocks recalled per question | 1.7 | 1.5 |

- **Latents + recall beat reading the whole document** (0.50 vs 0.41 EM)
  while reading ~1.7 raw blocks + latents, about a third of the tokens.
  This fits known effects: models get distracted by irrelevant context
  (Shi et al. 2023; HotpotQA's distractor paragraphs are chosen to be
  confusable), and both Coca-Cola and "you don't need long context" report
  compression or short contexts beating full context. Caveat: 0.50 vs 0.41
  is 3 questions of 32; it needs a larger eval to be firm. The gold blocks
  (0.63) show the ceiling.
- **Where recall loses:** coverage first (all gold blocks in the top 4
  only 63% of the time: HotpotQA needs two paragraphs, and evidence
  straddles blocks), then the choice (a gold block picked 81% of the time
  although one is always among the candidates). Next: paragraph-aligned
  blocks, a second-hop retrieval after the first recall, LLM yes/no
  reranking instead of entropy, a better embedding than mean pooling.
- **Recall ablation, 128 questions** (`recall_ablation`, 8 blocks, BM25,
  at most two units; culprit per wrong answer, F1 < 0.5 = wrong):

  | | EM | F1 | ok | reader | retrieval | choice | blocks read raw |
  |---|---|---|---|---|---|---|---|
  | full documents | 0.51 | 0.69 | | | | | 8 |
  | window only | 0.11 | 0.17 | | | | | 0 |
  | latents | 0.24 | 0.32 | | | | | 0 |
  | latents + gold blocks | 0.63 | 0.78 | | | | | |
  | V0 block units, entropy lookahead | 0.52 | 0.64 | 0.67 | 0.17 | 0.11 | 0.05 | 1.69 |
  | V1 paragraph units, entropy | 0.52 | 0.66 | **0.71** | 0.16 | 0.07 | 0.05 | 2.50 |
  | V2 paragraph, LLM yes/no rerank | 0.48 | 0.62 | 0.66 | 0.16 | 0.10 | 0.08 | 1.46 |
  | V3 paragraph, yes/no, second hop | 0.53 | 0.66 | 0.70 | 0.18 | **0.03** | 0.09 | 2.10 |
  | **V4 paragraph, second hop, entropy** | **0.57** | **0.71** | **0.76** | 0.16 | 0.04 | 0.04 | 2.72 |

  "reader" = wrong even with every gold block expanded (recall cannot fix
  it); "retrieval" = some evidence never a candidate; "choice" = evidence
  among the candidates but not all picked.
  - **With 128 questions, recall matches reading the whole document** (EM
    0.52–0.53 vs 0.51, F1 0.66 vs 0.69) from latents + ~2 raw blocks; the
    32-question "beats" was partly luck.
  - **The reader is the biggest limit (16–18%)**, then retrieval, which
    paragraph units (11% -> 7%) and a second hop (-> 3%) nearly remove; choice
    (5–9%) is now the main fixable loss.
  - **Asking the model "is this passage relevant?" chooses worse than the
    entropy lookahead** (choice errors 7–9% vs 5%).
  - **V4 (paragraph units + second hop + entropy) is the best recall:** 0.76
    ok of a 0.84 ceiling, retrieval and choice misses 4% each. Paired over
    the same 128 questions it beats V0 (EM +0.055, t 2.1; F1 +0.076, t 3.1)
    and V1 (EM +0.047, t 2.5), and it is at least as good as reading the
    whole document (EM +0.062, better on 17 questions, worse on 9, t 1.6;
    F1 +0.022, t 0.6) while reading the latents + 2.7 of 8 blocks raw.

**`chat_eval`, 32-block documents (~8k tokens), 32 questions** (EM, F1):
full 0.41 / 0.58 (thinking: 0.50 / 0.64), window only 0.06, latents 0.22,
question-conditioned latents 0.25, gold blocks 0.63 / 0.78, recall BM25
(V0 policy) 0.44 / 0.58, dense 0.38. Retrieval among 32 blocks: BM25 top-4
holds some gold block 94% of the time but all of them only 28% (63% at 8
blocks); mean-pooled dense retrieval collapses (66% / 0%). Recall still
matches the full document at ~1/8 of the tokens; full evidence coverage is
the part that degrades with length, which is what V3/V4's second hop
addresses.

**V4 at 32 blocks (~8k tokens), 128 questions** (`recall_ablation --blocks
32 --variants V0 V4`):

| | EM | F1 | ok | reader | retrieval | choice | blocks read raw |
|---|---|---|---|---|---|---|---|
| full documents (8.3k tokens) | 0.48 | 0.67 | | | | | 32 |
| window only | 0.14 | 0.19 | | | | | 0 |
| latents | 0.21 | 0.29 | | | | | 0 |
| latents + gold blocks | 0.64 | 0.79 | | | | | |
| V0 block + entropy | 0.47 | 0.60 | 0.66 | 0.13 | 0.19 | 0.02 | 1.7 |
| **V4 paragraph + second hop + entropy** | **0.53** | **0.67** | **0.72** | 0.13 | 0.08 | 0.07 | 2.7 |

- **V4 matches reading the whole 8k-token document** (paired: EM +0.047,
  better on 14 questions, worse on 8, t 1.3; F1 −0.008, a tie) from about
  1.3k positions (latents + 2.7 of 32 blocks + window), ~1/6 of the input,
  and beats V0 (EM +0.062, t 2.2; F1 +0.064, t 2.3).
- **The second hop matters more with length:** retrieval misses 19% -> 8%
  (8 blocks: 11% -> 4%). The remaining losses are spread out: reader 13%,
  retrieval 8%, choice 7%.
- **Thinking is confounded by the 256-token budget:** traces open with
  ~100 tokens of "Thinking Process: 1. Analyze the Request…", so answers
  are often forced mid-reasoning; even the oracle drops (0.50 vs 0.63).
  Latents fare worst with thinking (0.16 vs 0.28), consistent with the
  model reasoning them away.
- **Thinking-mode training** (`--think_prob 0.5`, `chat9b_think`): for
  half of the samples the frozen model reading the raw documents writes its
  reasoning (192 tokens, `</think>` forced if needed) and the student
  reading latents is trained to match the reasoning and the answer.
  Stopped at step 250 to save cost (~23 s per step).
- Practical: flash-linear-attention (fla 0.6.0) speeds the 9B step ~24%
  and cuts memory 48 -> 37 GB; causal-conv1d would need a CUDA devel build
  (skipped, it only replaces a 4-tap conv). Gradient checkpointing clashes
  with switching LoRA on and off by a module flag (the recomputation sees a
  different flag), so it is off.

**Thinking-mode training did not help (`chat9b_think`, stopped at step
250).** Half the samples trained on the teacher's reasoning (192 tokens).
On the same 32 questions (4 blocks) at step 250: latents with thinking
0.19 EM vs window-only 0.13, and latents without thinking fell to 0.16
(`chat9b` at step 250: 0.31); gold blocks + latents 0.47 (vs 0.59). Likely
cause: most trace tokens are document-independent boilerplate ("Thinking
Process: 1. Analyze the Request…"), so KL on them dilutes the signal. A
retry would weight the answer and the document-dependent parts of the
trace.

**Recall when the documents cannot fit (`long_eval`).** 512 blocks
(131k tokens; 8192 latents) per question, every mode within a 16k-position
budget (asserted). Stopped at 27 of 64 questions to save cost (one Modal
preemption on the way; the resumable JSONL recovered all 13 finished
questions):

| mode (16k budget) | EM | F1 | context positions |
|---|---|---|---|
| window only | 0.19 | 0.30 | 184 |
| truncate to the most recent 16k tokens | 0.22 | 0.34 | 16,312 |
| BM25 top-2 paragraphs, raw, no latents | 0.33 | 0.52 | 411 |
| BM25 paragraphs raw up to 14k tokens (RAG-fill) | 0.48 | 0.67 | 14,119 |
| latents only | 0.30 | 0.41 | 8,376 |
| latents + V0 recall | 0.44 | 0.56 | 8,696 |
| **latents + V4 recall** | **0.48** | **0.66** | 8,856 (~700 raw) |
| latents + gold blocks | 0.56 | 0.76 | 9,043 |

- **V4 matches the strongest RAG baseline** while reading about 20x less
  raw text (latents + ~2.7 blocks vs 14k tokens of retrieved paragraphs),
  and is far above truncation (what a plain 16k model would see).
- **Latents alone carry gist even 131k tokens back** (0.30 vs 0.19), well
  beyond the 1.3k-token span the compressor was trained on.
- **Compression dominates the cost:** ~50 s of ~79 s per question on an
  L40S go to compressing 512 blocks.
- RAG-fill does well because HotpotQA questions share words with their
  evidence. Benchmarks that would separate the approaches: chains whose
  next hop is not named in the question (RULER variable tracking),
  aggregation over the whole document, and memory with updated facts and
  paraphrased questions (LongMemEval).

**Where this leaves the idea.** The system does what it was designed for
once the reader is capable enough: on Qwen3.5-9B, latents give gist,
uncertainty-triggered lookahead recall with paragraph units and a second
hop finds the evidence, and the result matches reading everything (up to
8k tokens) or the best RAG baseline (131k tokens under a 16k budget) from
a fraction of the raw text. The small models never got there: with
Qwen3-0.6B on enwik8 continuation the latents kept almost nothing, though
objective, task and reader all changed together, so size alone is not
proven to be the cause. Open: thinking with latents, a learned retriever
for the remaining misses, streaming generation with recall mid-answer,
and benchmarks that RAG cannot solve.

What we learned:
- **Always measure the no-memory control.** "Student close to teacher"
  meant nothing here; the comparison that matters is student with vs
  without latents, and with the window removed.
- **A from-scratch byte model on enwik8 is a poor test bed for context
  compression;** a pretrained model with strong in-context retrieval gives
  a teacher worth imitating and tests (copy, passkey) with real signal.

