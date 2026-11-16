# Latent memory with recall

A way to give a language model a long memory without a long context window.
Old text is compressed into a few latent vectors per block that stay in the
context; the raw text is kept outside the context; when the model is unsure,
it recalls the raw text of the block that resolves its uncertainty.

The working version is in `rolling_latent_chat.py` (on top of
`rolling_latent_qwen.py`), trained and evaluated on Qwen3.5-9B with HotpotQA.
The full experimental history, including everything that failed, is in
`RESULTS.md` and `rolling_latent_lm.md`.

## 1. The idea: a cache hierarchy

| tier | what | where | cost |
|---|---|---|---|
| raw window | the most recent tokens, verbatim | context | 1 position per token |
| latents | every older block of 256 tokens as 16 vectors (16x) | context | 1 position per 16 tokens |
| raw store | all text, verbatim | outside (disk, RAM) | none in the context |

Latents are the cache, the raw store is the disk. Latents only need to carry
the *gist*: enough for the model to answer what it can and to know when it
cannot. Exact detail comes from recall, which copies a block (or paragraph)
from the raw store back into the context. Because the original is always
kept, nothing is ever lost.

This matches three published results: compressing context into latents
trained on the downstream decision (Coca-Cola, arXiv 2611.06500), keeping
contexts short with compaction plus selective reading of stored originals
("You don't need long context", arXiv 2611.05306), and the capacity of
input-embedding space (Kuratov et al. 2025, "Cramming 1568 tokens into a
single vector"). The closest prior work for recall is LCLM (arXiv 2606.09659),
whose agent expands compressed segments with a learned `expand` tool;
LatentPress (arXiv 2609.01507) independently uses the same frozen-reader,
CE + KL design without recall. What this design adds is a training-free
policy for *when* and *what* to expand (section 6).

## 2. Layout and positions

One sample, as the reader sees it (chat format, thinking off):

```
<|im_start|>user\n{instruction}<documents>\n        raw prefix         positions 0..P-1
[16 latents of block 0][16 of block 1] ...       latents            see below
[recent window: last W document tokens]          raw
\n</documents>\n\nQuestion: ...<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n    raw suffix
{answer}<|im_end|>                               generated / scored
```

- **Positions are the text's real positions, with gaps.** Block j covers
  document tokens `P + j*256 ... P + j*256 + 255`. Its 16 latents sit at
  the end of each 16-token sub-span: `P + j*256 + 15, + 31, ..., + 255`
  (`lat_pos`). Relative distances from later text to a latent then match
  the raw layout, which RoPE needs. An expanded block puts its raw tokens
  back at their own positions.
- **Always pass an explicit all-ones `attention_mask`.** In transformers 5.x
  a gap in `position_ids` with no mask is read as the start of a new packed
  sequence and attention across it is blocked, so nothing would see the
  latents (`RollingQwen.hidden`).
- Qwen3.5 is a hybrid (24 Gated DeltaNet linear-attention layers, 8
  full-attention layers, partial M-RoPE). The linear layers ignore positions;
  the gaps only matter in the full-attention layers. Install
  `flash-linear-attention` (~24% faster, 48 -> 37 GB at 9B).

## 3. The compressor

The same frozen model with LoRA switched on writes the latents
(`RollingQwen.compress`):

```
input  = [latents of the <= 3 previous blocks] [block j, 256 raw tokens]
         [conditioning: the next W document tokens, or the question] [16 learned beacon vectors]
h      = final hidden states at the 16 beacon positions (beacons share position max+1)
latent = RMSNorm(MLP(h)) * rms(token embeddings, detached) + type_vector        -> 16 x D
```

- **LoRA** (r 16, alpha 32) on every large linear layer: attention q/k/v/o
  or, in DeltaNet layers, `in_proj_qkv`, `in_proj_z`, `out_proj`; MLP
  gate/up/down. 74.8M trainable parameters at 9B, plus beacons, a D->D->D MLP
  adapter, the norm and a type vector.
- **Normalise the latents to token-embedding scale.** Unnormalised, they grew
  to 400–600x the embedding RMS and became content-free attention sinks.
- **Conditioning:** the next window of text (streaming: the question is not
  known yet) or the question (re-compression once the task is known). Mixed
  50/50 in training; in evaluation the two were equally good.
- Compression happens once per block, at eviction time, and is the main
  inference cost (~0.1 s per block at 9B on an L40S).

### Alternative: a lightweight writer (`--writer pool`)

Following LatentPress (arXiv 2609.01507): run the frozen reader's bottom 2
layers once over the whole document, apply an identity-initialised d x d
linear adapter (the only trained part, 16.8M parameters at d = 4096; no
LoRA), and mean-pool each 16-token sub-span into one latent at its end. It
is query-independent and roughly 100x cheaper to compress with than the
LoRA compressor; training takes 0.7–1.3 s per step and 24 GB instead of
4–6 s and 37 GB.

| 32 questions, 4 blocks, EM (F1) | LoRA compressor (step 250) | pool writer (step 500) |
|---|---|---|
| latents | 0.31 (0.34) | 0.22 (0.27) |
| latents + gold blocks | 0.59 (0.72) | 0.56 (0.67) |

Its latents carry less gist, but with recall it is nearly as good: V4
recall reaches 0.54 / 0.68 EM / F1 at 8 blocks (LoRA compressor: 0.57 /
0.71). At 131k tokens under a 16k budget, over 64 questions, it ties
RAG-fill (0.45 vs 0.42 EM) while compressing a document in 3 s instead of
50 s. It is the better default at scale.

## 4. The reader

The **frozen base model with LoRA off** reads the latents. It is exactly the
original model, so its behaviour on ordinary text is untouched and the
latents must adapt to it, not the other way round. LoRA is a module flag that
is flipped between the compressor (on) and the reader (off); gradient
checkpointing would recompute the reader's layers with the wrong flag, so it
is off (each question does its own backward instead; peak 37 GB at 9B).

## 5. Training

Per question (`chat_loss`):

```
teacher = frozen model, LoRA off, reading prefix + raw document + suffix + answer   (no grad)
latents = compressor over each document block (question-conditioned with p 0.5)
student = frozen model reading prefix + latents + window + suffix + answer
          (window dropped with p 0.25, so the latents cannot be ignored)
loss    = CE(student, gold answer tokens) + KL(teacher || student) on the answer tokens only
```

- **Train on the answer tokens only.** Training on generic text continuation
  (enwik8) taught the latents almost nothing: there, old context barely
  changes the next token. The answer to a question about the document is
  what has to survive compression.
- **Data:** HotpotQA (distractor). A document is the question's 10
  paragraphs (2 gold) shuffled with paragraphs from other questions until it
  fills 4 blocks of 256 tokens + a 128-token window; gold paragraphs always
  land in the compressed blocks. 30k training questions.
- **Settings:** batch 4 questions, AdamW lr 2e-4 (fused), warmup 50, cosine;
  ~4–6 s per step on an H100. The best checkpoint was early (step 250 of
  1000); later steps drifted slightly. Save every eval's checkpoint.

## 6. Recall at inference

V4 in `recall_ablation`, the best policy found:

```
units = paragraphs of the stored raw text (a paragraph expands every block it overlaps)
H0    = entropy of the reader's first answer token, with latents only
if H0 is low: answer from the latents
chosen = []
cand  = top-4 units by BM25(question)
repeat up to 2 times:
    if this is the second hop: cand = top-4 units by BM25(question + text of the first chosen unit)
    for u in cand: H(u) = first-token entropy with chosen + [u] expanded to raw text
    u* = argmin H(u);  stop if H(u*) >= current entropy;  chosen.append(u*)
answer = generate(prefix, latents with the chosen units' blocks raw, window, question)
```

- **Trigger = plain entropy of the first answer token.** AUROC 0.77 for "a
  block would help" (0.93 on copy-like content); a trained divergence head
  and TokUR's epistemic uncertainty (arXiv 2505.11737) were no better.
- **Choice = lookahead:** try each candidate and keep the one that makes the
  model most confident. It beat a trained pointer, attention-based pointers
  and asking the model "is this passage relevant? Yes/No".
- **Paragraph units** stop evidence from being split across blocks; the
  **second hop** finds the bridge fact that the question does not name.
- **Generation** is greedy and KV-cached with the exact gapped positions
  (`generate`); recall is decided once, before answering.
- **Retrieval:** BM25 works well on HotpotQA; mean-pooled hidden states make
  a poor dense retriever (all evidence among the top 4 only 31% of the
  time).

## 7. Results (Qwen3.5-9B, thinking off, compressor at step 1000)

| setting | full / best baseline | latents only | latents + V4 recall | latents + gold blocks |
|---|---|---|---|---|
| 8 blocks (~2k tokens), 128 q | full doc 0.51 EM / 0.69 F1 | 0.24 / 0.32 | **0.57 / 0.71** | 0.63 / 0.78 |
| 32 blocks (~8k tokens), 128 q | full doc 0.48 / 0.67 | 0.21 / 0.29 | **0.53 / 0.67** | 0.64 / 0.79 |
| 512 blocks (131k tokens), 16k budget, 27 q | RAG filled to 14k tokens 0.48 / 0.67; truncation 0.22 / 0.34 | 0.30 / 0.41 | **0.48 / 0.66** | 0.56 / 0.76 |

- Recall matches reading the whole document (paired over the same
  questions: EM +0.06, F1 +0.02 at 8 blocks; not significant against the
  full document, significant against the simpler recall policies) from the
  latents plus ~2.7 raw blocks. At 131k tokens it matches the strongest RAG
  baseline while reading ~20x less raw text, and the latents still carry
  gist although training never saw more than 1.3k tokens of history.
- Of the remaining errors at 8 blocks, 16% are the reader (wrong even with
  every gold block), 4% retrieval (evidence never a candidate) and 4%
  choice (`culprit` in `recall_ablation`).

## 8. Pitfalls we hit

| symptom | cause | fix |
|---|---|---|
| latents carry nothing; student = no-memory control | latents 400–600x embedding RMS; window explains everything | normalise latents; window dropout; always measure the no-memory control |
| latents get worse mid-training with a frozen reader | pointer/divergence heads' gradients reshape the latents | heads read detached states |
| nothing attends to the latents | transformers reads position gaps as packed-sequence boundaries | explicit all-ones `attention_mask` |
| `CheckpointError` in backward | LoRA on/off flag differs at recomputation | no gradient checkpointing |
| latents that regenerate a block exactly wreck the model as context | optimised for "recite this", ~3.3x embedding RMS | keep memory latents and exact recall separate; recall from raw text |
| `UR_RESULT_ERROR_DEVICE_LOST` on Intel Arc | PyTorch `_foreach_*` kernels | fused AdamW, `clip_grad_norm_(foreach=False)` |

## 9. What did not work

- **Small models and generic objectives:** a 26M byte model and Qwen3-0.6B
  trained on enwik8 continuation; their latents kept at most 0.02–0.05
  nats. The objective, task and reader all changed with the 9B, so model
  size alone is not proven to be the cause.
- **Latents as an archive:** per-block optimisation can pack 256 tokens into
  4 latents exactly (64x), and an encoder + ~100 optimisation steps gets
  there, but such latents are ~8x larger than the raw text in bytes, decode
  slower than re-reading, and are harmful as context.
- **Thinking with latents:** with thinking on, the model reasons that "only
  one snippet is provided" and ignores the latents (0.16 EM vs 0.28 without
  thinking); training on the teacher's reasoning traces (half the samples,
  250 steps) made both modes worse. Recall still works with thinking.
- **Trained pointer, attention pointers, yes/no reranking, TokUR** (see 6).

## 10. Running it

```
python rolling_latent_chat.py selftest                    # tiny random Qwen3.5, no download
modal run modal_rllm.py --script chat --gpu cpu --online --args "prep --out hotpot"
modal run --detach modal_rllm.py --script chat --gpu H100 --online \
    --args "train --data hotpot --out chat9b.pt --steps 250 --eval_every 250"
modal run --detach modal_rllm.py --script chat --gpu L40S \
    --args "recall_ablation --ckpt chat9b.pt --blocks 8 --q 128 --out_jsonl abl.jsonl"
modal run --detach modal_rllm.py --script chat --gpu L40S \
    --args "long_eval --ckpt chat9b.pt --blocks 512 --q 64 --out_jsonl long.jsonl"
modal run modal_rllm.py --script chat --gpu L40S --args "chat_demo --ckpt chat9b.pt --q 5"
```

Costs on Modal (Nov 2026): training 250 steps ~25 min on an H100 (~$2);
`recall_ablation` 128 questions ~45 min on an L40S (~$1.50); `long_eval`
~80 s per question at 512 blocks on an L40S (50 s of it compression). The
evals write one JSON line per question and resume after preemption.

## 11. Open problems

- Thinking with latents (weight the answer and document-dependent reasoning
  tokens; or recall during thinking).
- Recall during generation, not only before the first answer token.
- A learned retriever or better dense embedding for the remaining misses,
  and cheaper candidate selection over thousands of blocks.
- Benchmarks that keyword RAG cannot solve: chains whose next hop is not
  named in the question (RULER variable tracking), aggregation over the
  whole document, memory with updated facts (LongMemEval).
- A stronger reader: 16% of HotpotQA errors remain even with the gold
  evidence.
