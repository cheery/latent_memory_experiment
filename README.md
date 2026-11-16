# Latent memory with recall

A hybrid long-context memory for a frozen LLM. Older text is compressed, block by block, into a few
latent vectors that stay in the context window (16 latents per 256 tokens). The raw text is kept
outside the context. When the model is uncertain about its answer, it recalls the raw text of the
paragraph that resolves the uncertainty and puts it back into the context. Latents are the cache;
the raw store is the disk.

On Qwen3.5-9B with HotpotQA, this matches reading the whole document at 2k and 8k tokens, and
matches the strongest retrieval baseline on 131k-token documents under a 16k context budget, while
reading only the latents plus about 700 raw tokens.

This is a research experiment by an independent researcher, done with a few tens of dollars of
cloud GPU time. It is shared as a technical record, not as a finished paper.

## Is this worth publishing?

Worth putting out, yes. A full conference paper, not yet.

### Why it is worth putting out

- **It fills a gap that the closest work points at.** Memorilla (arXiv 2610.21375) finds that its
  latent memory and RAG solve different questions (on FactKG, 15.9% are solved only by Memorilla
  and 10.0% only by RAG), that their union beats both, and it explicitly names "hybrid retrieval and
  latent-memory systems" as future work. This is such a hybrid: latents for gist, plus
  uncertainty-triggered recall of the raw text.
- **The two closest papers have no recall.** Coca-Cola (arXiv 2611.06500) compresses tool
  observations into latents with no way back to the original. Memorilla produces a fixed set of
  query-conditioned memory tokens and doesn't recall either. "You don't need long context" (arXiv
  2611.05306) recalls raw text, but its compact state is text, not latents.
- **Several findings are useful to others, and some go against intuition:**
  - Latents that contain a block perfectly (they regenerate it exactly) can be harmful as memory,
    roughly doubling the reader's loss. Archive latents and context latents are different things.
  - Choosing which block to recall by "expand it and see whether the model becomes confident"
    (entropy lookahead) beats asking the model "is this passage relevant? Yes/No", a trained
    pointer, attention-based pointers, and TokUR's epistemic uncertainty.
  - Concrete failure modes, each with a fix: latents blowing up to hundreds of times the embedding
    scale and becoming attention sinks; recall heads silently degrading the latents through shared
    gradients; transformers 5.x treating gaps in `position_ids` as packed-sequence boundaries and
    blocking attention to the latents.
  - With thinking on, the model reasons the latents away ("only one snippet is provided"), while
    recall keeps working.
- **The headline result is solid as parity.** Recall matches reading the full document at 2k and 8k
  tokens, and the best RAG baseline at 131k tokens within a 16k budget, from a fraction of the raw
  text.

### Why it is not a full paper yet

- **The evaluation is thin:** one dataset (HotpotQA), 128 questions (27 at 131k tokens), one seed,
  and the advantage over reading the full document is not statistically significant. Memorilla
  evaluates on 8 benchmarks.
- **The baselines are weak:** BM25 RAG only. Missing: a strong dense retriever (for example
  Qwen3-Embedding), an Activation-Beacon- or ICAE-style compression baseline, and Memorilla itself on
  its own benchmarks.
- **The pieces have prior art,** which a paper must cite and position against:
  - triggering retrieval on model uncertainty: FLARE, Self-RAG;
  - choosing documents by how much they help the LM: REPLUG-style likelihood scoring;
  - rolling compression by the same model: AutoCompressors, Activation Beacon.

  The contribution is the combination (a cache hierarchy with latent gist plus raw recall) and the
  analysis, not any single new component.
- **There is no cost or latency comparison yet.** Compressing 512 blocks takes about 50 seconds per
  question on an L40S; a cost-versus-accuracy curve is needed.

### What would make it a solid paper

1. Benchmarks where keyword RAG struggles: chains whose next hop is not named in the question (RULER
   variable tracking), memory with updated facts (LongMemEval), and one benchmark shared with
   Memorilla (NarrativeQA or TriviaQA).
2. Stronger baselines: dense-retriever RAG, and a compression-only baseline.
3. Larger evaluations (500+ questions, 2–3 seeds) with confidence intervals.
4. A cost-versus-accuracy plot.

## Results

Qwen3.5-9B, frozen; compressor = the same model + LoRA, trained 1000 steps on HotpotQA answer tokens;
thinking off. EM / F1.

| setting | full document / best baseline | latents only | latents + recall | latents + gold blocks |
|---|---|---|---|---|
| 8 blocks (~2k tokens), 128 questions | full 0.51 / 0.69 | 0.24 / 0.32 | **0.57 / 0.71** | 0.63 / 0.78 |
| 32 blocks (~8k tokens), 128 questions | full 0.48 / 0.67 | 0.21 / 0.29 | **0.53 / 0.67** | 0.64 / 0.79 |
| 512 blocks (131k tokens), 16k budget, 27 questions | RAG filled to 14k tokens 0.48 / 0.67; truncation 0.22 / 0.34 | 0.30 / 0.41 | **0.48 / 0.66** | 0.56 / 0.76 |

"Recall" is the best policy found (V4): the entropy of the first answer token decides whether to
recall; BM25 over paragraphs proposes 4 candidates; each candidate is expanded in turn and the one
leaving the lowest entropy is kept; then a second hop retrieves again with the question plus what
was just read. At most two paragraphs are recalled.

## Files

| file | what |
|---|---|
| `latent_memory.md` | **Start here.** How the mechanism works and how to implement it: layout, compressor, reader, training, recall, pitfalls, what did not work, how to run. |
| `RESULTS.md` | Full experiment log in the order it happened, including the dead ends: a byte-level prototype, Qwen3-0.6B, latent capacity ("cramming"), recall decisions, and Qwen3.5-9B. |
| `rolling_latent_chat.py` | The working system: Qwen3.5-9B, chat-format HotpotQA, training (`train`), evaluation (`chat_eval`, `recall_ablation`, `long_eval`), and a demo that prints answers (`chat_demo`). |
| `rolling_latent_qwen.py` | The core model (`RollingQwen`: compressor, reader, latent positions, LoRA), plus the Qwen3-0.6B experiments, `recall_eval`, `needle_eval` and TokUR. |
| `cram.py` | Capacity experiments: how much text a frozen model regenerates from a few optimised vectors, and an encoder for them. |
| `rolling_latent_lm.py`, `rolling_latent_lm.md` | The original byte-level prototype and its design notes. It is kept as history; its latents collapsed. |
| `modal_rllm.py` | Runs any of the scripts on a Modal GPU, with data, model cache and checkpoints in a Modal volume. |

Every script has a `selftest` that runs on CPU with a tiny random model and needs no download:

```
pip install -r requirements.txt
python rolling_latent_chat.py selftest
python rolling_latent_qwen.py selftest
```

Training and evaluation need a GPU with about 40 GB (training) or about 24 GB (inference) for the 9B
model; see `latent_memory.md`, section 10, for the commands and costs.

## Related work

- Coca-Cola: latent context compression for agents, action-preserving training. arXiv 2611.06500.
- Memorilla: latent semantic memory for LLMs. arXiv 2610.21375.
- You don't need long context: compaction and selective reading. arXiv 2611.05306.
- TokUR: token-level uncertainty by low-rank weight perturbation. arXiv 2505.11737.
- Kuratov et al. 2025, "Cramming 1568 tokens into a single vector and back again". arXiv 2502.13063.
- AutoCompressors (Chevalier et al. 2023), ICAE (Ge et al. 2023), Activation Beacon, FLARE, Self-RAG,
  REPLUG: prior art for parts of the design.

## License

MIT, see `LICENSE`.
