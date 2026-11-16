#!/usr/bin/env python3
"""
rolling_latent_chat.py

Rolling latent compression on an instruction-tuned model (default Qwen/Qwen3.5-9B), tested on question answering
over long documents. Builds on rolling_latent_qwen.py (RollingQwen: compressor = base + LoRA, reader = the frozen
base model, latents RMS-normalised, recall heads not used here).

A sample is a chat turn:
    <|im_start|>user\\n{instruction}<documents>\\n  [ doc blocks ... | recent window ]  \\n</documents>\\n\\n
    Question: {q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n   {answer}<|im_end|>
The documents are HotpotQA paragraphs (the question's 10, gold and distractors) padded with paragraphs from other
questions up to n_blocks * block + window tokens; gold paragraphs always land in the compressible blocks.
    teacher : frozen model, everything raw
    student : frozen model reading [prefix raw | latents of the blocks | window + question raw]
    loss    : on the answer tokens only (Coca-Cola style): CE + w_kl * KL(teacher || student)
Latents of block j are compressed at eviction time from [earlier latents | block | conditioning], where the
conditioning is either the next `window` document tokens (streaming: the question is not known yet) or the question
itself (re-compression once the task is known), mixed in training by --qcond_prob.

Evaluation (chat_eval) on held-out HotpotQA questions, exact match / F1 of the generated answer, thinking off and on:
    full, none (window only), latents (streaming), latents_q (question-conditioned), oracle (gold blocks expanded),
    recall: entropy of the first answer token -> retrieval prefilter (BM25 or dense) -> lookahead (expand each
    candidate, keep the one leaving the lowest entropy), up to two recalls, then answer.

Commands:
    python rolling_latent_chat.py selftest                 # tiny random Qwen3.5, synthetic data, no download
    python rolling_latent_chat.py prep --out hotpot        # HotpotQA distractor -> hotpot_train.jsonl, hotpot_val.jsonl
    python rolling_latent_chat.py train --data hotpot --out chat9b.pt
    python rolling_latent_chat.py chat_eval --ckpt chat9b.pt --data hotpot --blocks 8 32
"""
import argparse
import json
import math
import os
import random
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn.functional as F

from rolling_latent_qwen import Config, RollingQwen, pick_device

INSTRUCTION = "Read the documents and answer the question with a short phrase.\n\n<documents>\n"


@dataclass
class ChatCfg:
    w_ce: float = 1.0
    w_kl: float = 1.0
    qcond_prob: float = 0.5        # share of samples whose blocks are compressed knowing the question
    drop_window_prob: float = 0.25  # share of samples where the student sees no raw document window
    think_prob: float = 0.0         # share of samples trained with thinking on: the teacher (raw documents) writes
                                    # its reasoning, the student must match it token by token, then the answer


# ============================================================================ data

def prep(args):
    """HotpotQA (distractor) parquet from the Hub -> JSON lines {q, a, paras: [[title, text]], gold: [titles]}."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    files = {"train": ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
             "val": ["distractor/validation-00000-of-00001.parquet"]}
    for split, names in files.items():
        n = 0
        with open(f"{args.out}_{split}.jsonl", "w") as f:
            for name in names:
                t = pq.read_table(hf_hub_download("hotpotqa/hotpot_qa", name, repo_type="dataset")).to_pylist()
                for r in t:
                    paras = [[ti, "".join(se)] for ti, se in zip(r["context"]["title"], r["context"]["sentences"])]
                    f.write(json.dumps(dict(q=r["question"], a=r["answer"], paras=paras,
                                            gold=sorted(set(r["supporting_facts"]["title"])), type=r["type"])) + "\n")
                    n += 1
                    if split == "train" and n >= args.max_train:
                        break
                if split == "train" and n >= args.max_train:
                    break
        print(f"{split}: {n} questions -> {args.out}_{split}.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def para_text(title, text):
    return f"Title: {title}\n{text}\n\n"


def build_doc(ex, pool, tok, n_blocks, Bk, W, rng):
    """Token ids of the documents (n_blocks * Bk + W) and the set of blocks that hold gold-paragraph tokens.
    The question's own paragraphs are shuffled in with filler paragraphs from other questions; gold ones are moved
    before the window if needed, so the evidence is always in compressed memory."""
    target = n_blocks * Bk + W
    items = [(tok.encode(para_text(t, x)), t in ex["gold"]) for t, x in ex["paras"]]
    total = sum(len(i) for i, _ in items)
    while total < target + 200:
        other = pool[rng.randrange(len(pool))]
        t, x = other["paras"][rng.randrange(len(other["paras"]))]
        ids = tok.encode(para_text(t, x))
        items.append((ids, False)); total += len(ids)
    rng.shuffle(items)
    for _ in range(20):                                                # gold paragraphs must end inside the blocks
        ends, pos = [], 0
        for ids, g in items:
            pos += len(ids); ends.append(pos)
        late = [k for k, (ids, g) in enumerate(items) if g and ends[k] > n_blocks * Bk]
        if not late:
            break
        k = late[0]
        items.insert(rng.randrange(0, max(1, k // 2)), items.pop(k))
    ids, gold_blocks, pos, spans = [], set(), 0, []
    for t_ids, g in items:
        if g:
            gold_blocks.update(range(pos // Bk, min(n_blocks, (pos + len(t_ids) - 1) // Bk + 1)))
        if pos < target:
            spans.append((pos, min(pos + len(t_ids), target), g))         # paragraph token spans (for recall units)
        ids += t_ids; pos += len(t_ids)
    return torch.tensor(ids[:target]), sorted(b for b in gold_blocks if b < n_blocks), spans


def chat_suffix(q, thinking):
    return (f"\n</documents>\n\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n"
            + ("<think>\n" if thinking else "<think>\n\n</think>\n\n"))


class Sample:
    """One question laid out for the model. Positions: prefix 0..P-1, document P.., then suffix and answer."""

    def __init__(self, ex, pool, tok, c: Config, n_blocks, rng, thinking=False, dev="cpu"):
        self.ex, self.n = ex, n_blocks
        Bk, W = c.block, c.window
        self.prefix = torch.tensor(tok.encode("<|im_start|>user\n" + INSTRUCTION), device=dev)
        doc, self.gold, spans = build_doc(ex, pool, tok, n_blocks, Bk, W, rng)
        self.paras = [(a, b, g) for a, b, g in spans if a < n_blocks * Bk]   # paragraphs starting in the blocks
        self.doc = doc.to(dev)
        self.P = len(self.prefix)
        self.q_ids = torch.tensor(tok.encode(f"Question: {ex['q']}"), device=dev)
        self.suffix = torch.tensor(tok.encode(chat_suffix(ex["q"], thinking)), device=dev)
        end = tok.encode("<|im_end|>")
        self.answer = torch.tensor(tok.encode(ex["a"]) + end, device=dev)
        self.Ld = n_blocks * Bk
        self.Bk, self.W = Bk, W

    def block(self, j):
        return self.doc[j * self.Bk:(j + 1) * self.Bk][None]

    def window(self):
        return self.doc[self.Ld:][None]

    def tail_start(self, with_window=True):
        return self.P + self.Ld + (0 if with_window else self.W)


def compress_doc(model, s: Sample, qcond):
    """Latents for every block, as at eviction time: each conditioned on <= n_blocks-1 earlier latents and on the
    next window of the document (streaming) or on the question (placed where it will appear)."""
    c = model.c
    lats = []
    for i in range(s.n):
        st = s.P + i * s.Bk
        prev = [(lats[j], s.P + j * s.Bk) for j in range(max(0, i - (c.n_blocks - 1)), i)]
        if qcond:
            cond, cpos = s.q_ids[None], s.tail_start() + 10
        else:
            cond, cpos = s.doc[(i + 1) * s.Bk:(i + 1) * s.Bk + s.W][None], st + s.Bk
        lats.append(model.compress(prev, s.block(i), st, cond, cpos))
    return lats


def parts_for(s: Sample, lats, expand=(), raw_all=False):
    parts = [("raw", s.prefix[None], 0)]
    for j in range(s.n):
        st = s.P + j * s.Bk
        if raw_all or j in expand:
            parts.append(("raw", s.block(j), st))
        elif lats is not None:                                         # lats None: block absent (window only)
            parts.append(("lat", lats[j], st))
    return parts


# ============================================================================ training

def answer_logits(model, parts, tail, tail_start, n_ans):
    """Logits predicting the last n_ans tokens of tail (the answer), with the reader."""
    h, _, off = model.student(parts, tail, tail_start)
    return model.logits(h[:, off + tail.shape[1] - n_ans - 1:off + tail.shape[1] - 1])


def teacher_trace(model, s: Sample, tok, budget):
    """Thinking sample: the frozen model reading the raw documents writes its reasoning (greedy, up to `budget`
    tokens, '</think>' forced if needed). Returns ids from the first thinking token through '</think>\n\n'."""
    tail = torch.cat([s.window()[0], s.suffix])[None]
    generate(model, parts_for(s, None, raw_all=True), tail, s.tail_start(), tok, think_budget=budget,
             stop_at_think_end=True)
    ids = generate.last_ids
    end = tok.encode("</think>")
    for k in range(len(ids) - len(end), -1, -1):                       # cut after the last '</think>'
        if ids[k:k + len(end)] == end:
            ids = ids[:k + len(end)]
            break
    return torch.tensor(ids + tok.encode("\n\n"), device=s.doc.device)


def chat_loss(model, s: Sample, cc: ChatCfg, qcond, drop_window, trace=None):
    """Teacher: everything raw (no grad). Student: latents. Scored: the assistant turn after the prompt, i.e. the
    answer, or with a thinking `trace` the reasoning + the answer. KL on all scored tokens, CE on the answer."""
    n_ans = len(s.answer)
    turn = s.answer if trace is None else torch.cat([trace, s.answer])
    n = len(turn)
    with torch.no_grad(), model.lora(False):
        full = torch.cat([s.prefix, s.doc, s.suffix, turn])[None]
        h = model.hidden(model.embed(full[:, :-1]), torch.arange(full.shape[1] - 1, device=full.device))
        lt = F.log_softmax(model.logits(h[:, -n:]), -1)
    lats = compress_doc(model, s, qcond)
    tail = torch.cat([s.window()[0], s.suffix, turn])[None] if not drop_window else torch.cat([s.suffix, turn])[None]
    lg = answer_logits(model, parts_for(s, lats), tail, s.tail_start(not drop_window), n)
    ls = F.log_softmax(lg, -1)
    y = turn[None]
    ce = F.nll_loss(ls[:, -n_ans:].flatten(0, 1), y[:, -n_ans:].flatten())
    kl = (lt.exp() * (lt - ls)).sum(-1).mean()
    return cc.w_ce * ce + cc.w_kl * kl, dict(ce_s=ce.item(), kl=kl.item(),
                                             ce_t=F.nll_loss(lt[:, -n_ans:].flatten(0, 1), y[:, -n_ans:].flatten()).item(),
                                             think=float(trace is not None))


def build_model(c: Config, dev):
    """No gradient checkpointing: LoRA is switched on (compressor) and off (reader) by a module flag, and
    checkpointing would recompute the reader's layers in backward with whatever the flag is then (CheckpointError).
    Each question does its own backward, so activations of one question (~15 GB at 9B) are the peak."""
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(c.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    return RollingQwen(c, lm.to(dev)).to(dev)


def train(args):
    from transformers import AutoTokenizer
    dev = pick_device()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    c = Config(model=args.model, block=args.block, k_lat=args.k_lat, n_blocks=args.n_blocks, window=args.window,
               future=0, lora_r=args.lora_r, reader_lora=False)
    cc = ChatCfg(**{f.name: getattr(args, f.name) for f in fields(ChatCfg)})
    tok = AutoTokenizer.from_pretrained(c.model)
    train_set, val_set = load_jsonl(args.data + "_train.jsonl"), load_jsonl(args.data + "_val.jsonl")
    model = build_model(c, dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"base {c.model} | trainable {sum(p.numel() for p in params) / 1e6:.1f}M | {len(train_set)} train questions"
          f" | {c.n_blocks} blocks x {c.block} tok -> {c.k_lat} latents each ({c.block // c.k_lat}x)", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=(dev != "cpu"))
    rng = random.Random(args.seed)
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 and step > 0 or step == args.steps:
            think_eval = cc.think_prob > 0
            r = chat_eval(model, tok, val_set, train_set, n_blocks=c.n_blocks, n_q=args.eval_q,
                          thinking_modes=(False, True) if think_eval else (False,), think_budget=args.think_budget,
                          modes=("full", "none", "latents", "oracle") if think_eval else
                          ("full", "none", "latents", "latents_q", "oracle"), verbose=False)
            print(f"== eval step {step}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float)),
                  flush=True)
            ck = {"trainable": model.trainable_state(), "config": asdict(c), "chat_cfg": asdict(cc), "step": step}
            torch.save(ck, args.out)
            torch.save(ck, os.path.splitext(args.out)[0] + f"_s{step}.pt")   # keep every eval's weights
            model.train()
            t0 = time.time()
        if step == args.steps:
            break
        f = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        logs = Counter()
        for _ in range(args.batch):                                    # micro-batches of one question
            thinking = rng.random() < cc.think_prob
            s = Sample(train_set[rng.randrange(len(train_set))], train_set, tok, c, c.n_blocks, rng,
                       thinking=thinking, dev=dev)
            trace = teacher_trace(model, s, tok, args.think_budget) if thinking else None
            loss, info = chat_loss(model, s, cc, rng.random() < cc.qcond_prob, rng.random() < cc.drop_window_prob,
                                   trace)
            (loss / args.batch).backward()
            logs.update({k: v / args.batch for k, v in info.items()})
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = (time.time() - t0) / (args.log_every if step else 1)
            t0 = time.time()
            mem = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev == "cuda" else ""
            print(f"step {step} | " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()) +
                  f" | gn {gn:.2f} | {dt:.2f} s/step{mem}", flush=True)


# ============================================================================ generation and evaluation

def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    em = float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return em, 0.0
    prec, rec = same / len(p), same / len(g)
    return em, 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate(model, parts, tail, tail_start, tok, max_new=32, think_budget=0, stop_at_think_end=False):
    """Greedy, KV-cached generation by the frozen reader with exact (gapped) positions. With think_budget > 0 the
    tail ends in '<think>\\n'; thinking runs until '</think>' or the budget, then the answer follows."""
    from transformers import DynamicCache
    lm = model.lm
    x, pos, _, _ = model.student_inputs(parts, tail, tail_start)
    dev = x.device
    end_id, think_end = tok.encode("<|im_end|>")[0], tok.encode("</think>")
    cache = DynamicCache(config=lm.model.config)                        # the text model's config (hybrid layers)
    nxt_pos = int(pos[-1]) + 1
    T = x.shape[1]

    def step(emb, p):
        nonlocal T
        out = lm.model(inputs_embeds=emb.to(model.dtype), position_ids=p[None], past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(1, T, dtype=torch.long, device=dev))
        return model.logits(out.last_hidden_state[:, -1])

    with model.lora(False):
        logits = step(x, pos)
        out, thinking = [], think_budget > 0
        budget = think_budget + max_new
        while len(out) < budget:
            nid = int(logits.argmax(-1))
            if thinking and len(out) >= think_budget and nid not in think_end:   # out of thinking budget
                forced = tok.encode("\n</think>\n\n")
                for fid in forced[:-1]:
                    out.append(fid); T += 1
                    step(model.embed(torch.tensor([[fid]], device=dev)), torch.tensor([nxt_pos], device=dev))
                    nxt_pos += 1
                nid, thinking = forced[-1], False
            out.append(nid)
            if nid == end_id:
                break
            if thinking and (nid == think_end[-1] if len(think_end) == 1 else tok.decode(out[-4:]).endswith("</think>")):
                thinking = False
                if stop_at_think_end:
                    break
            if stop_at_think_end and think_budget > 0 and not thinking:      # forced end of thinking
                break
            T += 1
            logits = step(model.embed(torch.tensor([[nid]], device=dev)), torch.tensor([nxt_pos], device=dev))
            nxt_pos += 1
    text = tok.decode(out)
    answer = text.split("</think>")[-1].replace("<|im_end|>", "").strip()
    generate.last_text, generate.last_ids = text, out                  # full output, thinking included
    return answer, len(out)


def first_token_entropy(model, parts, tail, tail_start):
    h, _, _ = model.student(parts, tail, tail_start)
    lp = F.log_softmax(model.logits(h[:, -1]), -1)
    return float(-(lp.exp() * lp).sum())


def bm25_rank(query, docs, k1=1.5, b=0.75):
    tokn = lambda t: re.findall(r"\w+", t.lower())
    D = [tokn(d) for d in docs]
    q = tokn(query)
    avg = np.mean([len(d) for d in D]) or 1
    df = Counter(w for d in D for w in set(d))
    N = len(D)
    scores = []
    for d in D:
        tf = Counter(d)
        sc = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                sc += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(sc)
    return list(np.argsort(scores)[::-1])


@torch.no_grad()
def dense_rank(model, s: Sample):
    """Mean final hidden state of each raw block (computed once at eviction) vs of the question, cosine."""
    with model.lora(False):
        emb = lambda ids: model.hidden(model.embed(ids[None]), torch.arange(ids.shape[0], device=ids.device))[0].float().mean(0)
        keys = torch.stack([emb(s.doc[j * s.Bk:(j + 1) * s.Bk]) for j in range(s.n)])
        qv = emb(s.q_ids)
    return F.cosine_similarity(keys, qv[None], -1).argsort(descending=True).tolist()


@torch.no_grad()
def chat_eval(model, tok, val_set, pool, n_blocks=8, n_q=64, thinking_modes=(False, True), think_budget=256,
              modes=("full", "none", "latents", "latents_q", "oracle", "recall"), retrievers=("bm25", "dense"),
              topk=4, max_recall=2, seed=1234, verbose=True):
    """Returns mean EM/F1 per mode (and per thinking setting), recall statistics and retrieval recall@k."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    rng = random.Random(seed)
    qs = [val_set[i] for i in rng.sample(range(len(val_set)), min(n_q, len(val_set)))]
    acc = {}
    stats = Counter()

    def add(key, pred, gold):
        em, f1 = em_f1(pred, gold)
        acc.setdefault(key + "_em", []).append(em); acc.setdefault(key + "_f1", []).append(f1)

    for qi, ex in enumerate(qs):
        for thinking in thinking_modes:
            r2 = random.Random(seed + qi)                              # same document for both thinking settings
            s = Sample(ex, pool, tok, c, n_blocks, r2, thinking=thinking, dev=dev)
            tail = torch.cat([s.window()[0], s.suffix])[None]
            ts = s.tail_start()
            tag = "think_" if thinking else ""
            budget = think_budget if thinking else 0
            lat_s = compress_doc(model, s, qcond=False) if any(m in modes for m in ("latents", "oracle", "recall")) else None
            for mode in modes:
                if mode == "full":
                    parts = parts_for(s, None, raw_all=True)
                elif mode == "none":
                    parts = parts_for(s, None)
                elif mode == "latents":
                    parts = parts_for(s, lat_s)
                elif mode == "latents_q":
                    parts = parts_for(s, compress_doc(model, s, qcond=True))
                elif mode == "oracle":
                    parts = parts_for(s, lat_s, expand=set(s.gold))
                elif mode == "recall":
                    continue
                pred, n_tok = generate(model, parts, tail, ts, tok, think_budget=budget)
                add(tag + mode, pred, ex["a"])
                stats[tag + mode + "_tokens"] += n_tok
            if "recall" in modes:
                # decide from the non-thinking first answer token, as "check memory before answering"
                dtail = torch.cat([s.window()[0], torch.tensor(tok.encode(chat_suffix(ex["q"], False)), device=dev)])[None]
                ent0 = first_token_entropy(model, parts_for(s, lat_s), dtail, ts)
                stats[tag + "entropy_sum"] += ent0
                block_text = [tok.decode(s.doc[j * s.Bk:(j + 1) * s.Bk].tolist()) for j in range(s.n)]
                for ret in retrievers:
                    order = bm25_rank(ex["q"], block_text) if ret == "bm25" else dense_rank(model, s)
                    cand = [int(j) for j in order[:topk]]
                    stats[f"{tag}{ret}_recall@{topk}"] += float(any(g in cand for g in s.gold))
                    stats[f"{tag}{ret}_allgold@{topk}"] += float(all(g in cand for g in s.gold))
                    chosen, best = [], ent0
                    for _ in range(max_recall):                         # greedy lookahead, keep only if it helps
                        trials = [(first_token_entropy(model, parts_for(s, lat_s, expand=set(chosen + [j])), dtail, ts), j)
                                  for j in cand if j not in chosen]
                        if not trials:
                            break
                        e, j = min(trials)
                        if e >= best:
                            break
                        chosen.append(j); best = e
                    stats[f"{tag}{ret}_recalls"] += len(chosen)
                    stats[f"{tag}{ret}_hit"] += float(any(j in s.gold for j in chosen))
                    pred, n_tok = generate(model, parts_for(s, lat_s, expand=set(chosen)), tail, ts, tok,
                                           think_budget=budget)
                    add(f"{tag}recall_{ret}", pred, ex["a"])
                    stats[f"{tag}recall_{ret}_tokens"] += n_tok
                    acc.setdefault(f"{tag}recall_{ret}_entropy", []).append(ent0)   # for an offline tau sweep
            if verbose and qi < 3:
                print(f"  Q: {ex['q']} | gold: {ex['a']} | gold blocks {s.gold}")
    n = len(qs)
    res = {k: float(np.mean(v)) for k, v in acc.items() if not k.endswith("_entropy")}
    res.update({k: v / n for k, v in stats.items()})
    res["questions"], res["n_blocks"] = n, n_blocks
    model.train()
    return res


@torch.no_grad()
def recall_blocks(model, s: Sample, lats, tok, retriever="dense", topk=4, max_recall=2):
    """The recall policy of chat_eval for one question: candidates from the retriever, greedy lookahead on the
    first answer token's entropy, keep a block only if it lowers it. Returns (chosen blocks, entropy before)."""
    dev = s.doc.device
    dtail = torch.cat([s.window()[0], torch.tensor(tok.encode(chat_suffix(s.ex["q"], False)), device=dev)])[None]
    ts = s.tail_start()
    ent0 = first_token_entropy(model, parts_for(s, lats), dtail, ts)
    if retriever == "bm25":
        order = bm25_rank(s.ex["q"], [tok.decode(s.doc[j * s.Bk:(j + 1) * s.Bk].tolist()) for j in range(s.n)])
    else:
        order = dense_rank(model, s)
    cand, chosen, best = [int(j) for j in order[:topk]], [], ent0
    for _ in range(max_recall):
        trials = [(first_token_entropy(model, parts_for(s, lats, expand=set(chosen + [j])), dtail, ts), j)
                  for j in cand if j not in chosen]
        if not trials:
            break
        e, j = min(trials)
        if e >= best:
            break
        chosen.append(j); best = e
    return chosen, ent0


@torch.no_grad()
def chat_demo(model, tok, val_set, pool, n_q=5, n_blocks=8, think_budget=256, seed=99):
    """Print what the model actually answers for a few questions in every mode (thinking off), and with thinking
    on for full / latents / recall (answer + the start of the reasoning)."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    rng = random.Random(seed)
    for qi in rng.sample(range(len(val_set)), n_q):
        ex = val_set[qi]
        for thinking in (False, True):
            s = Sample(ex, pool, tok, c, n_blocks, random.Random(seed + qi), thinking=thinking, dev=dev)
            if not thinking:
                print(f"\nQ: {ex['q']}\n   gold: {ex['a']}   | evidence in blocks {s.gold} of {n_blocks}")
            tail = torch.cat([s.window()[0], s.suffix])[None]
            ts = s.tail_start()
            lat = compress_doc(model, s, qcond=False)
            chosen, ent0 = recall_blocks(model, s, lat, tok)
            modes = [("full", parts_for(s, None, raw_all=True)), ("none", parts_for(s, None)),
                     ("latents", parts_for(s, lat))]
            if not thinking:
                modes += [("latents_q", parts_for(s, compress_doc(model, s, qcond=True))),
                          ("oracle", parts_for(s, lat, expand=set(s.gold)))]
            modes.append((f"recall {chosen} (entropy {ent0:.2f})", parts_for(s, lat, expand=set(chosen))))
            for name, parts in modes:
                pred, _ = generate(model, parts, tail, ts, tok, think_budget=think_budget if thinking else 0)
                mark = "✓" if em_f1(pred, ex["a"])[1] >= 0.5 else " "
                if thinking:
                    thought = generate.last_text.split("</think>")[0].strip().replace("\n", " ")
                    print(f"   [think {name}] {mark} {pred!r}\n        thought: {thought[:220]!r}...")
                else:
                    print(f"   [{name}] {mark} {pred!r}")
    model.train()


# ============================================================================ recall ablation (what limits recall)

@dataclass
class RecallCfg:
    name: str
    unit: str = "block"        # "block": a 256-token block; "paragraph": a paragraph, expanding every block it covers
    rerank: str = "entropy"    # "entropy": lookahead on the first answer token; "yesno": P(Yes) of the reader judging
                               # the raw passage alone ("does this passage help answer ...?")
    hops: int = 1              # 2: after the first unit, re-retrieve with the question + the first unit's text
    topk: int = 4
    max_units: int = 2


RECALL_VARIANTS = [RecallCfg("V0_block_entropy"),                                 # = chat_eval's recall_bm25
                   RecallCfg("V1_para_entropy", unit="paragraph"),
                   RecallCfg("V2_para_yesno", unit="paragraph", rerank="yesno"),
                   RecallCfg("V3_para_yesno_2hop", unit="paragraph", rerank="yesno", hops=2),
                   RecallCfg("V4_para_entropy_2hop", unit="paragraph", hops=2)]   # V3's retrieval, V1's choice


def units_of(s: Sample, unit):
    """Recall units as (blocks to expand, token span) pairs."""
    if unit == "block":
        return [([j], (j * s.Bk, (j + 1) * s.Bk)) for j in range(s.n)]
    return [(list(range(a // s.Bk, min(s.n, (b - 1) // s.Bk + 1))), (a, b)) for a, b, _ in s.paras]


@torch.no_grad()
def yes_prob(model, tok, passage, question, known=None):
    """The frozen reader judges a raw passage on its own: P(Yes) / (P(Yes) + P(No))."""
    dev = next(model.parameters()).device
    ctx = f"Known so far:\n{known}\n\n" if known else ""
    prompt = (f"<|im_start|>user\n{ctx}Passage:\n{passage}\n\nDoes this passage contain information needed to answer "
              f"the question: {question}\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = torch.tensor(tok.encode(prompt), device=dev)
    with model.lora(False):
        lg = model.logits(model.hidden(model.embed(ids[None]), torch.arange(len(ids), device=dev))[:, -1])[0]
    y, n = tok.encode("Yes")[0], tok.encode("No")[0]
    return float(torch.softmax(lg[[y, n]].float(), -1)[0])


@torch.no_grad()
def recall_v(model, s: Sample, lats, tok, rc: RecallCfg):
    """One recall variant. Returns (blocks to expand, blocks that were ever candidates)."""
    dev = s.doc.device
    units = units_of(s, rc.unit)
    text = [tok.decode(s.doc[a:b].tolist()) for _, (a, b) in units]
    q = s.ex["q"]
    dtail = torch.cat([s.window()[0], torch.tensor(tok.encode(chat_suffix(q, False)), device=dev)])[None]
    ts = s.tail_start()
    chosen, cand_blocks = [], set()
    ent = lambda blocks: first_token_entropy(model, parts_for(s, lats, expand=set(blocks)), dtail, ts)
    blocks_of = lambda us: sorted({b for u in us for b in units[u][0]})
    if rc.rerank == "entropy":
        cand = [int(u) for u in bm25_rank(q, text)[:rc.topk]]
        cand_blocks.update(blocks_of(cand))
        best = ent([])
        for hop in range(rc.max_units):
            if hop == 1 and rc.hops == 2 and chosen:                   # second hop: re-retrieve with what was read
                cand = [int(u) for u in bm25_rank(q + " " + text[chosen[0]], text) if int(u) not in chosen][:rc.topk]
                cand_blocks.update(blocks_of(cand))
            trials = [(ent(blocks_of(chosen + [u])), u) for u in cand if u not in chosen]
            if not trials:
                break
            e, u = min(trials)
            if e >= best:
                break
            chosen.append(u); best = e
        return blocks_of(chosen), sorted(cand_blocks)
    # yes/no reranking
    cand = [int(u) for u in bm25_rank(q, text)[:rc.topk]]
    cand_blocks.update(blocks_of(cand))
    scored = sorted(((yes_prob(model, tok, text[u], q), u) for u in cand), reverse=True)
    chosen.append(scored[0][1])
    if rc.hops == 1:
        if len(scored) > 1 and scored[1][0] > 0.5 and rc.max_units > 1:
            chosen.append(scored[1][1])
    else:
        known = text[chosen[0]]
        cand2 = [int(u) for u in bm25_rank(q + " " + known, text) if int(u) not in chosen][:rc.topk]
        cand_blocks.update(blocks_of(cand2))
        s2 = sorted(((yes_prob(model, tok, text[u], q, known=known), u) for u in cand2), reverse=True)
        if s2 and s2[0][0] > 0.5:
            chosen.append(s2[0][1])
    return blocks_of(chosen), sorted(cand_blocks)


def culprit(correct, oracle_correct, gold, cand, chosen):
    if correct:
        return "ok"
    if not oracle_correct:
        return "reader"               # wrong even with every gold block expanded: recall cannot fix it
    if not set(gold) <= set(cand):
        return "retrieval"            # some evidence was never a candidate
    if not set(gold) <= set(chosen):
        return "choice"               # all evidence was among the candidates, not all of it was picked
    return "other"                    # all evidence expanded and still wrong


@torch.no_grad()
def recall_ablation(model, tok, val_set, pool, n_blocks=8, n_q=128, variants=RECALL_VARIANTS, out_jsonl=None,
                    seed=1234):
    """Per question: full / none / latents / oracle answers and every recall variant, with the culprit of each
    recall failure. Results are appended to out_jsonl question by question; a rerun skips questions already there
    (resume). V0 reproduces chat_eval's recall_bm25 policy and is the in-run baseline (the question sample
    differs from chat_eval's, which draws a different n_q)."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    rng = random.Random(seed)
    qidx = rng.sample(range(len(val_set)), min(n_q, len(val_set)))
    done = {}
    if out_jsonl and os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            r = json.loads(line)
            if r["n_blocks"] == n_blocks:
                done[r["qi"]] = r                                      # variants missing from a record are added below
    if done:
        print(f"resuming: {len(done)} of {len(qidx)} questions already in {out_jsonl} (missing variants are added)",
              flush=True)
    f = open(out_jsonl, "a") if out_jsonl else None
    names = [rc.name for rc in variants]
    for k, qi in enumerate(qidx):
        if qi in done and all(nm in done[qi] for nm in names):
            continue
        ex = val_set[qi]
        s = Sample(ex, pool, tok, c, n_blocks, random.Random(seed + k), dev=dev)
        tail = torch.cat([s.window()[0], s.suffix])[None]
        ts = s.tail_start()
        lat = compress_doc(model, s, qcond=False)
        r = done.get(qi) or dict(qi=qi, n_blocks=n_blocks, q=ex["q"], gold_answer=ex["a"], gold_blocks=s.gold)
        for mode, parts in (("full", parts_for(s, None, raw_all=True)), ("none", parts_for(s, None)),
                            ("latents", parts_for(s, lat)), ("oracle", parts_for(s, lat, expand=set(s.gold)))):
            if mode in r:
                continue
            pred, _ = generate(model, parts, tail, ts, tok)
            r[mode] = dict(pred=pred, em=em_f1(pred, ex["a"])[0], f1=em_f1(pred, ex["a"])[1])
        ok_ = lambda m: m["f1"] >= 0.5
        for rc in variants:
            if rc.name in r:
                continue
            chosen, cand = recall_v(model, s, lat, tok, rc)
            pred, _ = generate(model, parts_for(s, lat, expand=set(chosen)), tail, ts, tok)
            em, f1 = em_f1(pred, ex["a"])
            r[rc.name] = dict(pred=pred, em=em, f1=f1, chosen=chosen, cand=cand,
                              culprit=culprit(f1 >= 0.5, ok_(r["oracle"]), s.gold, cand, chosen))
        done[qi] = r
        if f:                                                          # a record per (question, update); the last
            f.write(json.dumps(r) + "\n"); f.flush()                    # one for a question wins when reloading
        if (k + 1) % 16 == 0:
            print(f"  {k + 1}/{len(qidx)} questions", flush=True)
    if f:
        f.close()
    recs = [done[qi] for qi in qidx if qi in done]
    model.train()
    return summarize_ablation(recs, [rc.name for rc in variants])


def summarize_ablation(recs, names):
    n = len(recs)
    res = dict(questions=n, n_blocks=recs[0]["n_blocks"] if recs else 0)
    for m in ["full", "none", "latents", "oracle"] + list(names):
        res[m + "_em"] = float(np.mean([r[m]["em"] for r in recs]))
        res[m + "_f1"] = float(np.mean([r[m]["f1"] for r in recs]))
    for m in names:
        cnt = Counter(r[m]["culprit"] for r in recs)
        for k in ("ok", "reader", "retrieval", "choice", "other"):
            res[f"{m}_{k}"] = cnt[k] / n
        res[m + "_units"] = float(np.mean([len(r[m]["chosen"]) for r in recs]))
    return res


def print_ablation(res, names):
    n = res["questions"]
    print(f"\n{n} questions, {res['n_blocks']} blocks per document (culprit: F1 < 0.5 counted as wrong)")
    print(f"{'mode':>22} {'EM':>6} {'F1':>6}")
    for m in ["full", "none", "latents", "oracle"] + list(names):
        print(f"{m:>22} {res[m + '_em']:>6.3f} {res[m + '_f1']:>6.3f}")
    print(f"\n{'recall variant':>22} {'ok':>6} {'reader':>7} {'retriev':>8} {'choice':>7} {'other':>6} {'blocks':>7}")
    for m in names:
        print(f"{m:>22} " + " ".join(f"{res[f'{m}_{k}']:>{w}.3f}" for k, w in
                                     (("ok", 6), ("reader", 7), ("retrieval", 8), ("choice", 7), ("other", 6)))
              + f" {res[m + '_units']:>7.2f}")


# ============================================================================ long documents under a context budget

@torch.no_grad()
def long_eval(model, tok, val_set, pool, n_blocks=512, n_q=64, budget=16384, rag_fill_tokens=14000,
              out_jsonl=None, seed=4321):
    """Documents of n_blocks * block tokens (512 x 256 = 131k) that cannot be read raw: every mode must fit in
    `budget` positions (asserted). Modes: none (window), truncate (most recent raw text that fits), rag2 (BM25
    top-2 paragraphs raw, no latents), rag_fill (BM25 paragraphs raw up to rag_fill_tokens, no latents), latents,
    V0 recall, V4 recall, oracle (gold blocks). Resumable JSONL like recall_ablation."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    rng = random.Random(seed)
    qidx = rng.sample(range(len(val_set)), min(n_q, len(val_set)))
    done = {}
    if out_jsonl and os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            r = json.loads(line)
            if r["n_blocks"] == n_blocks:
                done[r["qi"]] = r
        print(f"resuming: {len(done)} of {len(qidx)} questions done", flush=True)
    f = open(out_jsonl, "a") if out_jsonl else None
    v0 = next(rc for rc in RECALL_VARIANTS if rc.name == "V0_block_entropy")
    v4 = next(rc for rc in RECALL_VARIANTS if rc.name == "V4_para_entropy_2hop")
    for k, qi in enumerate(qidx):
        if qi in done:
            continue
        ex = val_set[qi]
        t0 = time.time()
        s = Sample(ex, pool, tok, c, n_blocks, random.Random(seed + k), dev=dev)
        tail = torch.cat([s.window()[0], s.suffix])[None]
        ts = s.tail_start()
        lat = compress_doc(model, s, qcond=False)
        r = dict(qi=qi, n_blocks=n_blocks, q=ex["q"], gold_answer=ex["a"], gold_blocks=s.gold,
                 doc_tokens=int(s.Ld + s.W), compress_s=round(time.time() - t0, 1))
        para_text = [tok.decode(s.doc[a:b].tolist()) for a, b, _ in s.paras]
        order = [int(u) for u in bm25_rank(ex["q"], para_text)]

        def raw_paras(us):                                           # paragraphs as raw parts, in document order
            return [("raw", s.doc[a:b][None], s.P + a) for a, b, _ in sorted(s.paras[u] for u in us)]

        fixed = len(s.prefix) + tail.shape[1] + 48                     # prefix, window, question, answer room
        n_trunc = max(0, (budget - fixed) // s.Bk)
        fill, used = [], 0
        for u in order:
            a, b, _ = s.paras[u]
            if used + (b - a) > rag_fill_tokens:
                break
            fill.append(u); used += b - a
        modes = {
            "none": parts_for(s, None),
            "truncate": [("raw", s.prefix[None], 0)] + [("raw", s.block(j), s.P + j * s.Bk)
                                                         for j in range(n_blocks - n_trunc, n_blocks)],
            "rag2": [("raw", s.prefix[None], 0)] + raw_paras(order[:2]),
            "rag_fill": [("raw", s.prefix[None], 0)] + raw_paras(fill),
            "latents": parts_for(s, lat),
            "oracle": parts_for(s, lat, expand=set(s.gold)),
        }
        for rc in (v0, v4):
            chosen, cand = recall_v(model, s, lat, tok, rc)
            modes[rc.name] = parts_for(s, lat, expand=set(chosen))
            r[rc.name + "_chosen"], r[rc.name + "_cand"] = chosen, cand
        for name, parts in modes.items():
            size = sum(p[1].shape[1] for p in parts) + tail.shape[1]
            assert size + 48 <= budget, (name, size)                   # nothing may exceed the context budget
            pred, _ = generate(model, parts, tail, ts, tok)
            em, f1 = em_f1(pred, ex["a"])
            r[name] = dict(pred=pred, em=em, f1=f1, ctx=size)
        r["truncate_has_gold"] = any(g >= n_blocks - n_trunc for g in s.gold)
        r["rag_fill_gold_paras"] = sum(1 for u in fill if s.paras[u][2])      # gold paragraphs in the RAG fill
        r["seconds"] = round(time.time() - t0, 1)
        done[qi] = r
        if f:
            f.write(json.dumps(r) + "\n"); f.flush()
        print(f"  {k + 1}/{len(qidx)}: {r['seconds']} s | " + " ".join(
            f"{m}={'✓' if r[m]['f1'] >= 0.5 else '·'}" for m in modes), flush=True)
    if f:
        f.close()
    recs = [done[qi] for qi in qidx if qi in done]
    model.train()
    return summarize_long(recs)


def summarize_long(recs):
    names = ["none", "truncate", "rag2", "rag_fill", "latents", "V0_block_entropy", "V4_para_entropy_2hop", "oracle"]
    res = dict(questions=len(recs), n_blocks=recs[0]["n_blocks"], doc_tokens=float(np.mean([r["doc_tokens"] for r in recs])))
    for m in names:
        res[m] = (float(np.mean([r[m]["em"] for r in recs])), float(np.mean([r[m]["f1"] for r in recs])),
                  float(np.mean([r[m]["ctx"] for r in recs])))
    for v in ("V0_block_entropy", "V4_para_entropy_2hop"):
        res[v + "_hit_all"] = float(np.mean([set(r["gold_blocks"]) <= set(r[v + "_chosen"]) for r in recs]))
        res[v + "_cand_all"] = float(np.mean([set(r["gold_blocks"]) <= set(r[v + "_cand"]) for r in recs]))
    res["truncate_has_gold"] = float(np.mean([r["truncate_has_gold"] for r in recs]))
    return res


def print_long(res):
    print(f"\n{res['questions']} questions, documents of {res['doc_tokens']:.0f} tokens ({res['n_blocks']} blocks)")
    print(f"{'mode':>22} {'EM':>6} {'F1':>6} {'context':>8}")
    for m in ["none", "truncate", "rag2", "rag_fill", "latents", "V0_block_entropy", "V4_para_entropy_2hop", "oracle"]:
        em, f1, ctx = res[m]
        print(f"{m:>22} {em:>6.3f} {f1:>6.3f} {ctx:>8.0f}")
    print(f"  all gold among candidates / expanded: V0 {res['V0_block_entropy_cand_all']:.3f} / "
          f"{res['V0_block_entropy_hit_all']:.3f} | V4 {res['V4_para_entropy_2hop_cand_all']:.3f} / "
          f"{res['V4_para_entropy_2hop_hit_all']:.3f} | truncation window holds gold {res['truncate_has_gold']:.3f}")


def print_chat_report(res):
    print(f"\n{res['questions']} HotpotQA questions, {res['n_blocks']} blocks per document")
    rows = sorted({k[:-3] for k in res if k.endswith("_em")})
    print(f"{'mode':>22} {'EM':>6} {'F1':>6} {'gen tokens':>10}")
    for r in rows:
        print(f"{r:>22} {res[r + '_em']:>6.3f} {res[r + '_f1']:>6.3f} {res.get(r + '_tokens', float('nan')):>10.1f}")
    other = {k: v for k, v in res.items() if not k.endswith(("_em", "_f1", "_tokens")) and k not in ("questions", "n_blocks")}
    print("  " + " | ".join(f"{k} {v:.3f}" for k, v in sorted(other.items())))


# ============================================================================ selftest

def selftest():
    """Tiny random Qwen3.5 (hybrid linear/full attention), synthetic HotpotQA-like data, word-level fake tokenizer."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    torch.manual_seed(0); random.seed(0)
    dev = pick_device()
    ok = lambda m: print(f"  [ok] {m}")
    qc = Qwen3_5TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                           linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=8192)
    lm = Qwen3_5ForCausalLM(qc).float()
    lm.config._attn_implementation = "sdpa"
    c = Config(model="tiny", block=32, k_lat=4, n_blocks=3, window=16, future=0, lora_r=4, reader_lora=False)
    model = RollingQwen(c, lm.to(dev)).to(dev)

    class Tok:                                                         # word-level, special tokens kept whole
        vocab = {}
        @classmethod
        def encode(cls, t):
            return [cls.vocab.setdefault(w, len(cls.vocab) % 500 + 5) for w in re.findall(r"<\|[a-z_]+\|>|</?think>|\S+", t)]
        @classmethod
        def decode(cls, ids):
            inv = {v: k for k, v in cls.vocab.items()}
            return " ".join(inv.get(i, "?") for i in ids)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma".split()
    mk = lambda i: dict(q=f"what is {words[i % 18]} ?", a=words[(i + 3) % 18],
                        paras=[[f"T{i}_{k}", " ".join(random.choice(words) for _ in range(12))] for k in range(4)],
                        gold=[f"T{i}_0", f"T{i}_1"])
    data = [mk(i) for i in range(40)]

    s = Sample(data[0], data, Tok, c, 3, random.Random(0), dev=dev)
    assert s.doc.shape[0] == 3 * 32 + 16 and all(0 <= g < 3 for g in s.gold) and len(s.gold) >= 1
    ok(f"documents: {s.doc.shape[0]} tokens, gold paragraphs inside the compressed blocks {s.gold}")

    cc = ChatCfg()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    for qcond in (False, True):
        for nowin in (False, True):
            loss, info = chat_loss(model, s, cc, qcond, nowin)
            assert torch.isfinite(loss), (qcond, nowin)
            opt.zero_grad(); loss.backward()
            assert model.beacons.grad.abs().sum() > 0 and any(lo.B.grad.abs().sum() > 0 for lo in model.loras)
    ok(f"chat loss finite with/without question conditioning and window; compressor gets gradients ({info})")
    st = Sample(data[1], data, Tok, c, 3, random.Random(1), thinking=True, dev=dev)
    tr = teacher_trace(model, st, Tok, budget=5)
    assert Tok.decode(tr.tolist()).split()[-1] == "</think>" or "</think>" in Tok.decode(tr.tolist())
    loss, info = chat_loss(model, st, cc, False, False, tr)
    assert torch.isfinite(loss) and info["think"] == 1.0
    opt.zero_grad(); loss.backward()
    assert model.beacons.grad.abs().sum() > 0
    ok(f"thinking sample: teacher trace of {len(tr)} tokens ends in </think>, KL over trace + answer, gradients flow")
    first = None
    for _ in range(60):
        loss, info = chat_loss(model, s, cc, False, False)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first, (first, loss.item())
    ok(f"loss falls on a fixed sample: {first:.3f} -> {loss.item():.3f}")

    with torch.no_grad():                                              # KV-cached generation == full recompute
        lats = compress_doc(model, s, False)
        parts = parts_for(s, lats)
        tail = torch.cat([s.window()[0], s.suffix])[None]
        pred, n = generate(model, parts, tail, s.tail_start(), Tok, max_new=6)
        seq = tail
        ref = []
        for _ in range(n):
            h, _, _ = model.student(parts, seq, s.tail_start())
            nid = int(model.logits(h[:, -1]).argmax(-1))
            ref.append(nid)
            seq = torch.cat([seq, torch.tensor([[nid]], device=dev)], 1)
        ref_text = Tok.decode(ref).split("</think>")[-1].replace("<|im_end|>", "").strip()
        assert ref_text == pred, (ref_text, pred)
    ok("KV-cached generation with gapped positions matches step-by-step recomputation")

    res = chat_eval(model, Tok, data[:4], data, n_blocks=3, n_q=2, thinking_modes=(False, True), think_budget=4,
                    verbose=False)
    assert res["questions"] == 2 and all(0 <= res[k] <= 1 for k in res if k.endswith(("_em", "_f1")))
    print_chat_report(res)
    ok("chat_eval runs in every mode, thinking off and on (budget-forced </think>)")
    chat_demo(model, Tok, data[:4], data, n_q=1, n_blocks=3, think_budget=4)
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "abl.jsonl")
    r1 = recall_ablation(model, Tok, data[:6], data, n_blocks=3, n_q=2, out_jsonl=path)
    r2 = recall_ablation(model, Tok, data[:6], data, n_blocks=3, n_q=3, out_jsonl=path)    # resumes: 2 done, 1 new
    assert r1["questions"] == 2 and r2["questions"] == 3 and len(open(path).readlines()) == 3
    extra = RECALL_VARIANTS + [RecallCfg("Vx_test", unit="paragraph", hops=2, topk=2)]
    r3 = recall_ablation(model, Tok, data[:6], data, n_blocks=3, n_q=3, variants=extra, out_jsonl=path)
    assert "Vx_test_em" in r3 and len(open(path).readlines()) == 6     # each question updated once with Vx_test
    names = [rc.name for rc in RECALL_VARIANTS]
    assert all(abs(sum(r2[f"{m}_{k}"] for k in ("ok", "reader", "retrieval", "choice", "other")) - 1) < 1e-9 for m in names)
    print_ablation(r2, names)
    ok("recall_ablation: all variants, culprits sum to 1, resumes from its JSONL and adds missing variants")
    path2 = os.path.join(tempfile.mkdtemp(), "long.jsonl")
    rl = long_eval(model, Tok, data[:6], data, n_blocks=12, n_q=2, budget=320, rag_fill_tokens=60, out_jsonl=path2)
    print_long(rl)
    assert rl["questions"] == 2 and all(rl[m][2] <= 320 for m in ("truncate", "rag_fill", "oracle", "latents"))
    ok("long_eval: every mode fits the context budget, truncation / RAG baselines, resumable")
    ok("chat_demo prints every mode")
    assert em_f1("The Eiffel Tower", "eiffel tower") == (1.0, 1.0) and em_f1("Paris, France", "Paris")[1] > 0.6
    ok("EM/F1 normalisation")
    print("SELFTEST PASSED")


# ============================================================================ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("prep")
    p.add_argument("--out", default="hotpot")
    p.add_argument("--max_train", type=int, default=30000)
    t = sub.add_parser("train")
    t.add_argument("--data", default="hotpot")
    t.add_argument("--out", default="chat.pt")
    t.add_argument("--model", default="Qwen/Qwen3.5-9B")
    t.add_argument("--block", type=int, default=256)
    t.add_argument("--k_lat", type=int, default=16)
    t.add_argument("--n_blocks", type=int, default=4)
    t.add_argument("--window", type=int, default=128)
    t.add_argument("--lora_r", type=int, default=16)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--eval_every", type=int, default=250)
    t.add_argument("--eval_q", type=int, default=32)
    t.add_argument("--log_every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--think_budget", type=int, default=192, help="teacher reasoning tokens for thinking samples")
    for f in fields(ChatCfg):
        t.add_argument(f"--{f.name}", type=float, default=f.default)
    d = sub.add_parser("chat_demo", help="print the answers for a few questions in every mode")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--data", default="hotpot")
    d.add_argument("--q", type=int, default=5)
    d.add_argument("--blocks", type=int, default=8)
    d.add_argument("--think_budget", type=int, default=256)
    le = sub.add_parser("long_eval", help="131k-token documents under a 16k context budget (resumable)")
    le.add_argument("--ckpt", required=True)
    le.add_argument("--data", default="hotpot")
    le.add_argument("--blocks", type=int, default=512)
    le.add_argument("--q", type=int, default=64)
    le.add_argument("--budget", type=int, default=16384)
    le.add_argument("--out_jsonl", default="long_eval.jsonl")
    ra = sub.add_parser("recall_ablation", help="recall variants + what limits each (resumable)")
    ra.add_argument("--ckpt", required=True)
    ra.add_argument("--data", default="hotpot")
    ra.add_argument("--blocks", type=int, default=8)
    ra.add_argument("--q", type=int, default=128)
    ra.add_argument("--out_jsonl", default="recall_ablation.jsonl")
    ra.add_argument("--variants", nargs="+", default=None, help="subset of variant names (default: all)")
    e = sub.add_parser("chat_eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", default="hotpot")
    e.add_argument("--blocks", type=int, nargs="+", default=[8, 32])
    e.add_argument("--q", type=int, default=64)
    e.add_argument("--think_budget", type=int, default=256)
    e.add_argument("--no_thinking_eval", action="store_true")
    e.add_argument("--topk", type=int, default=4)
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "prep":
        prep(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "long_eval":
        from transformers import AutoTokenizer
        ck = torch.load(args.ckpt, map_location=pick_device())
        c = Config(**ck["config"])
        model = build_model(c, pick_device())
        model.load_state_dict(ck["trainable"], strict=False)
        tok = AutoTokenizer.from_pretrained(c.model)
        val = load_jsonl(args.data + "_val.jsonl")
        print_long(long_eval(model, tok, val, val, n_blocks=args.blocks, n_q=args.q, budget=args.budget,
                             out_jsonl=args.out_jsonl))
    elif args.cmd == "recall_ablation":
        from transformers import AutoTokenizer
        ck = torch.load(args.ckpt, map_location=pick_device())
        c = Config(**ck["config"])
        model = build_model(c, pick_device())
        model.load_state_dict(ck["trainable"], strict=False)
        tok = AutoTokenizer.from_pretrained(c.model)
        val = load_jsonl(args.data + "_val.jsonl")
        vs = [rc for rc in RECALL_VARIANTS if args.variants is None or rc.name in args.variants]
        assert vs, f"no variant matches {args.variants}"
        res = recall_ablation(model, tok, val, val, n_blocks=args.blocks, n_q=args.q, out_jsonl=args.out_jsonl,
                              variants=vs)
        print_ablation(res, [rc.name for rc in vs])
    elif args.cmd == "chat_demo":
        from transformers import AutoTokenizer
        ck = torch.load(args.ckpt, map_location=pick_device())
        c = Config(**ck["config"])
        model = build_model(c, pick_device())
        model.load_state_dict(ck["trainable"], strict=False)
        tok = AutoTokenizer.from_pretrained(c.model)
        val = load_jsonl(args.data + "_val.jsonl")
        chat_demo(model, tok, val, val, n_q=args.q, n_blocks=args.blocks, think_budget=args.think_budget)
    else:
        from transformers import AutoTokenizer
        ck = torch.load(args.ckpt, map_location=pick_device())
        c = Config(**ck["config"])
        model = build_model(c, pick_device())
        model.load_state_dict(ck["trainable"], strict=False)
        tok = AutoTokenizer.from_pretrained(c.model)
        val, pool = load_jsonl(args.data + "_val.jsonl"), load_jsonl(args.data + "_val.jsonl")
        out = []
        for nb in args.blocks:
            res = chat_eval(model, tok, val, pool, n_blocks=nb, n_q=args.q, think_budget=args.think_budget,
                            thinking_modes=(False,) if args.no_thinking_eval else (False, True), topk=args.topk)
            print_chat_report(res)
            out.append(res)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
