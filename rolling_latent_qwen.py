#!/usr/bin/env python3
"""
rolling_latent_qwen.py

Rolling latent context compression on a pretrained Qwen3 base model (token-level port of
rolling_latent_lm.py). Older context is compressed block by block into K latent vectors; the model with
[latents | recent window | future] is trained to behave like the frozen base model reading the full raw
context [old blocks | recent window | future].

    teacher    : frozen base model, raw context, LoRA off                          (no gradients)
    compressor : base + LoRA reads [earlier latents | block | conditioning window | K beacons];
                 beacon outputs -> MLP adapter -> RMS-normalised to token-embedding scale -> K latents
    student    : base + LoRA reads [latents (one block may be raw = "expanded") | window | future]

Losses on the future region only (latents may encode the window, which is their conditioning):
    ce_s : student next-token CE        kl  : KL(teacher || student)
    div  : head predicting log1p(KL)    ptr : on copy probes, pointer over blocks -> block holding the copied span
Batch modes: copy probes (future = verbatim span of the old region), expanded block, re-conditioning on the
final window, window dropout (student sees [latents | future] only, so it cannot ignore the latents).

Lessons carried over from the byte-level prototype: unnormalised latents grew to ~500x embedding RMS and
became content-free attention sinks; with a raw window available the student learned to ignore the latents.
Hence lat_norm and window dropout here, and `no_window_*` metrics in eval.

Commands:
    python rolling_latent_qwen.py selftest                      # tiny random Qwen3, no download
    python rolling_latent_qwen.py prep --data enwik8 --out enwik8.qwen3.npy
    python rolling_latent_qwen.py train --tokens enwik8.qwen3.npy --out q06.pt
    python rolling_latent_qwen.py eval --ckpt q06.pt --tokens enwik8.qwen3.npy
"""
import argparse
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    d = os.environ.get("RLLM_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    return "xpu" if torch.xpu.is_available() else "cpu"


@dataclass
class Config:
    model: str = "Qwen/Qwen3-0.6B-Base"
    block: int = 256        # tokens per compressible block
    k_lat: int = 16         # latents per block -> compression block/k_lat
    n_blocks: int = 4       # old blocks per sample
    window: int = 256       # recent raw window
    future: int = 256       # scored region
    lora_r: int = 16
    lora_alpha: float = 32.0
    reader_lora: bool = False  # False: the student (reader of latents) is the untouched base model, as in
                               # Coca-Cola; LoRA only shapes the compressor. True: the q06_3k behaviour.
    ptr_dim: int = 128


@dataclass
class LossCfg:
    w_ce: float = 1.0
    w_kl: float = 1.0
    w_div: float = 0.1
    w_ptr: float = 0.5
    probe_cue: int = 16     # copied tokens shown before copy accuracy / pointer are scored
    detach_prev: bool = False
    head_grad: bool = False   # False: div head and pointer read DETACHED hidden states, so they learn to read the
                              # latents without reshaping them. With a frozen reader (q06_frozen) their gradients
                              # otherwise flowed into the compressor and made the latents hurt the LM (-0.13 nats)


# ============================================================================ model

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank update in fp32. B starts at zero, so the model starts as the base model."""

    def __init__(self, base: nn.Linear, r, alpha):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.perturb = None         # TokUR: (U' (out, r'), eps (in, r')) adds U' eps^T to the weight; see TokUR

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)
        if self.perturb is not None:
            U, eps = self.perturb
            y = y + ((x.float() @ eps) @ U.t()).to(y.dtype)
        return y


class RollingQwen(nn.Module):
    def __init__(self, c: Config, lm):
        super().__init__()
        self.c, self.lm = c, lm
        for p in lm.parameters():
            p.requires_grad_(False)
        self.loras = []
        for layer in lm.model.layers:
            # Qwen3: self_attn + mlp in every layer. Qwen3.5: full-attention layers have self_attn, Gated DeltaNet
            # layers have linear_attn (in_proj_qkv, in_proj_z, out_proj; the per-head in_proj_a/b are left alone)
            for pname in ("self_attn", "linear_attn", "mlp"):
                parent = getattr(layer, pname, None)
                if parent is None:
                    continue
                for n, ch in list(parent.named_children()):
                    if isinstance(ch, nn.Linear) and min(ch.in_features, ch.out_features) >= lm.get_input_embeddings().weight.shape[1] // 16:
                        lo = LoRALinear(ch, c.lora_r, c.lora_alpha)
                        setattr(parent, n, lo)
                        self.loras.append(lo)
        D = lm.get_input_embeddings().weight.shape[1]          # (VLM configs keep hidden_size in text_config)
        dev = lm.get_input_embeddings().weight.device
        emb = lm.get_input_embeddings().weight
        self.register_buffer("emb_rms", emb.float().pow(2).mean().sqrt().detach())
        self.beacons = nn.Parameter(torch.randn(c.k_lat, D, device=dev) * float(self.emb_rms))
        self.adapter = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(dev)
        self.lat_out_norm = RMSNorm(D).to(dev)
        self.lat_type = nn.Parameter(torch.zeros(D, device=dev))
        self.div_head = nn.Linear(D, 1).to(dev)
        self.ptr_q = nn.Linear(D, c.ptr_dim, bias=False).to(dev)
        self.ptr_k = nn.Linear(D, c.ptr_dim, bias=False).to(dev)
        for m in (self.adapter, self.div_head, self.ptr_q, self.ptr_k):
            for p in m.parameters():
                nn.init.normal_(p, std=0.02) if p.dim() > 1 else nn.init.zeros_(p)

    @property
    def dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    @contextmanager
    def lora(self, on):
        prev = [lo.enabled for lo in self.loras]
        for lo in self.loras:
            lo.enabled = on
        try:
            yield
        finally:
            for lo, p in zip(self.loras, prev):
                lo.enabled = p

    def embed(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def hidden(self, x, pos, attentions=False):
        """x: (B,T,D) input embeddings, pos: (T,) positions (increasing, gaps allowed) -> final-norm hidden.
        The explicit all-ones attention_mask matters: without it, transformers treats every gap in position_ids
        as the start of a new packed sequence and blocks attention across it, so nothing could see the latents.
        attentions=True: also return per-layer attention maps (B,H,T,T), computed with eager attention."""
        B, T = x.shape[:2]
        kw = dict(inputs_embeds=x.to(self.dtype), position_ids=pos[None].expand(B, -1),
                  attention_mask=torch.ones(B, T, dtype=torch.long, device=x.device), use_cache=False)
        if not attentions:
            return self.lm.model(**kw).last_hidden_state
        prev = self.lm.config._attn_implementation
        self.lm.set_attn_implementation("eager")
        try:
            out = self.lm.model(**kw, output_attentions=True)
        finally:
            self.lm.set_attn_implementation(prev)
        return out.last_hidden_state, out.attentions

    def logits(self, h):
        return self.lm.lm_head(h).float()

    def lat_pos(self, start, device):
        s = self.c.block // self.c.k_lat
        return start + torch.arange(self.c.k_lat, device=device) * s + (s - 1)

    def compress(self, prev, block_ids, block_start, cond_ids, cond_start):
        """prev: list of (latents (B,K,D) fp32, block_start). -> latents (B,K,D) fp32."""
        dev, B, K = block_ids.device, block_ids.shape[0], self.c.k_lat
        xs, ps = [], []
        for lat, st in prev:
            xs.append(lat.to(self.dtype))
            ps.append(self.lat_pos(st, dev))
        xs.append(self.embed(block_ids))
        ps.append(block_start + torch.arange(block_ids.shape[1], device=dev))
        if cond_ids is not None and cond_ids.shape[1] > 0:
            xs.append(self.embed(cond_ids))
            ps.append(cond_start + torch.arange(cond_ids.shape[1], device=dev))
        pos = torch.cat(ps)
        xs.append(self.beacons.to(self.dtype).unsqueeze(0).expand(B, -1, -1))
        pos = torch.cat([pos, torch.full((K,), int(pos.max()) + 1, device=dev, dtype=torch.long)])
        h = self.hidden(torch.cat(xs, 1), pos)[:, -K:].float()
        return self.lat_out_norm(self.adapter(h)) * self.emb_rms + self.lat_type

    def student(self, parts, tail, tail_start, attentions=False):
        """parts: list of ('lat', (B,K,D), start) | ('raw', (B,L) ids, start). -> hidden, seg (T,), offset of tail
        (+ attention maps if attentions). Runs with LoRA only if c.reader_lora: by default the reader is the
        frozen base model."""
        with self.lora(self.c.reader_lora):
            return self._student(parts, tail, tail_start, attentions)

    def _student(self, parts, tail, tail_start, attentions=False):
        x, pos, seg, off = self.student_inputs(parts, tail, tail_start)
        if attentions:
            h, att = self.hidden(x, pos, attentions=True)
            return h, seg, off, att
        return self.hidden(x, pos), seg, off

    def student_inputs(self, parts, tail, tail_start):
        """The student's input embeddings (B,T,D), positions (T,), part index per position (T,), tail offset."""
        dev = tail.device
        xs, ps, seg = [], [], []
        for j, (kind, t, st) in enumerate(parts):
            if kind == "lat":
                xs.append(t.to(self.dtype))
                p = self.lat_pos(st, dev)
            else:
                xs.append(self.embed(t))
                p = st + torch.arange(t.shape[1], device=dev)
            ps.append(p)
            seg.append(torch.full_like(p, j))
        off = sum(x.shape[1] for x in xs)
        xs.append(self.embed(tail))
        p = tail_start + torch.arange(tail.shape[1], device=dev)
        ps.append(p)
        seg.append(torch.full_like(p, -1))
        return torch.cat(xs, 1), torch.cat(ps), torch.cat(seg), off

    def block_summaries(self, h, seg, n_parts):
        return torch.stack([h[:, seg == j].float().mean(1) for j in range(n_parts)], 1)

    def pointer_scores(self, summaries, hq):
        q, k = self.ptr_q(hq.float()), self.ptr_k(summaries)
        return q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])


def build(c: Config, dev, lm=None):
    if lm is None:
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(c.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    return RollingQwen(c, lm.to(dev)).to(dev)


# ============================================================================ data and losses

def sample_len(c):
    return c.n_blocks * c.block + c.window + c.future


def make_batch(tokens, c: Config, bsz, probe, device):
    """tokens: 1-D int tensor. Copy probes: the whole future region is a verbatim span of the old region."""
    L_old, L = c.n_blocks * c.block, sample_len(c)
    ix = torch.randint(0, len(tokens) - L - 1, (bsz,))
    x = torch.stack([tokens[i:i + L] for i in ix.tolist()]).long()
    src = None
    if probe:
        src = torch.randint(0, L_old - c.future + 1, (bsz,))
        for b in range(bsz):
            x[b, L - c.future:] = x[b, src[b]:src[b] + c.future]
        src = src.to(device)
    return x.to(device), src


def compute(model, x, src, lc: LossCfg, expand_idx=None, recond=False, memory=True, drop_window=False):
    """Returns losses and intermediates. memory=False: no latents (eval control)."""
    assert memory or not drop_window
    c, dev = model.c, x.device
    L_old, W, Fu, Bk = c.n_blocks * c.block, c.window, c.future, c.block
    L = x.shape[1]
    out = {}
    with torch.no_grad(), model.lora(False):
        h_t = model.hidden(model.embed(x[:, :-1]), torch.arange(L - 1, device=dev))
        logit_tf = model.logits(h_t[:, L - Fu - 1:])                  # predicts x[:, L-Fu:]
    y_f = x[:, L - Fu:]

    lats = []
    for i in range(c.n_blocks if memory else 0):
        bs = i * Bk
        cs = L_old if recond else bs + Bk
        prev = [((l.detach() if lc.detach_prev else l), j * Bk) for j, l in enumerate(lats)]
        lats.append(model.compress(prev, x[:, bs:bs + Bk], bs, x[:, cs:cs + W], cs))
    parts = ([("raw", x[:, j * Bk:(j + 1) * Bk], j * Bk) if j == expand_idx else ("lat", lats[j], j * Bk)
              for j in range(c.n_blocks)] if memory else [])
    Wk = 0 if drop_window else W
    h_s, seg, off = model.student(parts, x[:, L_old + W - Wk:L - 1], L_old + W - Wk)
    s_idx = off + Wk - 1 + torch.arange(Fu, device=dev)
    hs_f = h_s[:, s_idx]
    logit_s = model.logits(hs_f)

    out["ce_s"] = F.cross_entropy(logit_s.flatten(0, 1), y_f.flatten())
    out["ce_t"] = F.cross_entropy(logit_tf.flatten(0, 1), y_f.flatten())
    lt, ls = F.log_softmax(logit_tf, -1), F.log_softmax(logit_s, -1)
    kl_pos = (lt.exp() * (lt - ls)).sum(-1)
    out["kl"] = kl_pos.mean()
    hd = h_s if lc.head_grad else h_s.detach()                         # what the recall heads read
    div_pred = model.div_head(hd[:, s_idx].float()).squeeze(-1)
    out["div"] = (div_pred - torch.log1p(kl_pos.detach().clamp_min(0))).pow(2).mean()
    out["loss"] = lc.w_ce * out["ce_s"] + lc.w_kl * out["kl"] + lc.w_div * out["div"]

    if src is not None and memory:
        m = torch.arange(lc.probe_cue, Fu, device=dev)
        summ = model.block_summaries(hd, seg, c.n_blocks)
        plog = model.pointer_scores(summ, hd[:, off + Wk - 1 + m])      # query = position predicting token m
        tgt = (src[:, None] + m[None]) // Bk
        out["ptr"] = F.cross_entropy(plog.flatten(0, 1), tgt.flatten())
        out["loss"] = out["loss"] + lc.w_ptr * out["ptr"]
        out["ptr_logits"], out["ptr_tgt"] = plog, tgt
    out.update(logit_s=logit_s, logit_tf=logit_tf, y_f=y_f, kl_pos=kl_pos, div_pred=div_pred)
    return out


# ============================================================================ evaluation

def passkey_batch(tokens, c: Config, tok, rng):
    """Filler from the data; a 5-digit pass key stated inside a random old block; the recent window ends with
    the question, so the key's tokens are the first future tokens. Returns x (1,L), block index, answer length."""
    L_old, L = c.n_blocks * c.block, sample_len(c)
    key = str(rng.randint(10000, 99999))
    needle = tok.encode(f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n")
    question = tok.encode("\nWhat is the pass key? The pass key is")
    answer = tok.encode(f" {key}")
    i = rng.randint(0, len(tokens) - L - 1)
    x = tokens[i:i + L].clone().long()
    j = rng.randrange(c.n_blocks)
    at = j * c.block + rng.randint(0, c.block - len(needle))
    x[at:at + len(needle)] = torch.tensor(needle)
    q0 = L_old + c.window - len(question)
    x[q0:L_old + c.window] = torch.tensor(question)
    x[L_old + c.window:L_old + c.window + len(answer)] = torch.tensor(answer)
    return x[None], j, len(answer)


@torch.no_grad()
def evaluate(model, tokens, lc: LossCfg, n_batches=10, bsz=4, n_probes=32, tok=None, n_passkey=32):
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    r = {}
    acc = {k: [] for k in ("ce_t", "ce_s", "ce_no_memory", "ce_no_window", "ce_no_window_no_memory", "kl")}
    preds, acts = [], []
    for _ in range(n_batches):
        x, _ = make_batch(tokens, c, bsz, False, dev)
        o = compute(model, x, None, lc)
        acc["ce_t"].append(o["ce_t"].item()); acc["ce_s"].append(o["ce_s"].item()); acc["kl"].append(o["kl"].item())
        acc["ce_no_memory"].append(compute(model, x, None, lc, memory=False)["ce_s"].item())
        # no window: score future tokens 1.. for both, so the no-memory variant has a predecessor token
        o_nw = compute(model, x, None, lc, drop_window=True)
        acc["ce_no_window"].append(F.cross_entropy(o_nw["logit_s"][:, 1:].flatten(0, 1), x[:, -c.future + 1:].flatten()).item())
        L_old, W = c.n_blocks * c.block, c.window
        h, _, _ = model.student([], x[:, L_old + W:-1], L_old + W)
        acc["ce_no_window_no_memory"].append(
            F.cross_entropy(model.logits(h).flatten(0, 1), x[:, L_old + W + 1:].flatten()).item())
        preds.append(o["div_pred"].flatten()); acts.append(torch.log1p(o["kl_pos"].clamp_min(0)).flatten())
    for k, v in acc.items():
        r[k if k == "kl" else k + "_nats"] = float(np.mean(v))
    p, a = torch.cat(preds), torch.cat(acts)
    cc = float(torch.corrcoef(torch.stack([p, a]))[0, 1])
    r["div_corr"] = cc if math.isfinite(cc) else 0.0
    r["div_pred_natural"] = float(p.mean())

    cue = lc.probe_cue
    copy = {k: [] for k in ("teacher", "no_memory", "none", "oracle", "pointer")}
    hits_tr, hits_any, div_probe = [], [], []

    def cacc(logits, y):
        return (logits[:, cue:].argmax(-1) == y[:, cue:]).float().mean().item()

    for _ in range(n_probes):
        x, src = make_batch(tokens, c, 1, True, dev)
        o = compute(model, x, src, lc)
        copy["teacher"].append(cacc(o["logit_tf"], o["y_f"]))
        copy["none"].append(cacc(o["logit_s"], o["y_f"]))
        copy["no_memory"].append(cacc(compute(model, x, src, lc, memory=False)["logit_s"], o["y_f"]))
        div_probe.append(o["div_pred"][:, cue:].mean().item())
        oracle = int((src[0] + cue + (c.future - cue) // 2) // c.block)
        choice = int(o["ptr_logits"][0, 0].argmax())
        first, last_b = int((src[0] + cue) // c.block), int((src[0] + c.future - 1) // c.block)
        hits_tr.append(float(choice == first)); hits_any.append(float(first <= choice <= last_b))
        copy["oracle"].append(cacc(compute(model, x, src, lc, expand_idx=oracle)["logit_s"], o["y_f"]))
        copy["pointer"].append(cacc(compute(model, x, src, lc, expand_idx=choice)["logit_s"], o["y_f"]))
    for k, v in copy.items():
        r[f"copy_acc_{k}"] = float(np.mean(v))
    r["pointer_hit_trained"], r["pointer_hit_any"] = float(np.mean(hits_tr)), float(np.mean(hits_any))
    r["div_pred_probe"] = float(np.mean(div_probe))

    if tok is not None and n_passkey:
        rng = random.Random(1234)
        pk = {k: [] for k in ("teacher", "no_memory", "none", "oracle")}
        for _ in range(n_passkey):
            x, j, n = passkey_batch(tokens, c, tok, rng)
            x = x.to(dev)
            o = compute(model, x, None, lc)

            def exact(logits):
                return float((logits[0, :n].argmax(-1) == o["y_f"][0, :n]).all())
            pk["teacher"].append(exact(o["logit_tf"])); pk["none"].append(exact(o["logit_s"]))
            pk["no_memory"].append(exact(compute(model, x, None, lc, memory=False)["logit_s"]))
            pk["oracle"].append(exact(compute(model, x, None, lc, expand_idx=j)["logit_s"]))
        for k, v in pk.items():
            r[f"passkey_{k}"] = float(np.mean(v))
    model.train()
    return r


# ============================================================================ recall decisions

class TokUR:
    """Training-free token-level uncertainty (Zhang et al., "TokUR", ICLR 2026). W_Q and W_K of every attention
    layer get a low-rank perturbation W = W0 + U' eps^T, U' = top-r' left singular vectors of W0, eps ~ N(0, sigma^2)
    (paper defaults r'=8, sigma=0.1, M=2). Over M perturbed passes:
        TU = H(mean_m p_m)   AU = mean_m H(p_m)   EU = TU - AU  (epistemic: mutual information)."""

    def __init__(self, model, r=8, sigma=0.1, M=2):
        self.sigma, self.M, self.mods = sigma, M, []
        for layer in model.lm.model.layers:
            attn = getattr(layer, "self_attn", None)                   # Qwen3.5: only the full-attention layers
            if attn is None:
                continue
            for lo in (attn.q_proj, attn.k_proj):
                U = torch.linalg.svd(lo.base.weight.float(), full_matrices=False)[0][:, :r].contiguous()
                self.mods.append((lo, U))

    @contextmanager
    def sample(self):
        for lo, U in self.mods:
            lo.perturb = (U, torch.randn(lo.base.in_features, U.shape[1], device=U.device) * self.sigma)
        try:
            yield
        finally:
            for lo, _ in self.mods:
                lo.perturb = None

    @torch.no_grad()
    def __call__(self, logits_fn):
        """logits_fn() -> (B,T,V) logits. Returns TU, AU, EU, each (B,T), in nats."""
        ps = []
        for _ in range(self.M):
            with self.sample():
                ps.append(F.softmax(logits_fn().float(), -1))
        ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        tu = ent(torch.stack(ps).mean(0))
        au = torch.stack([ent(p) for p in ps]).mean(0)
        return tu, au, tu - au


def auroc(score, label):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])                                   # average ranks over ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def cram_blocks(model, blocks, steps=400, lr=1e-2):
    """Phase-A latents for each block (cram.py): optimise K vectors per block so the FROZEN reader, given only
    [latents], regenerates the block. blocks: (n, Bk) ids. Latents sit at the student's lat_pos offsets and the
    block text follows, so the relative geometry matches how the student later reads them (RoPE is relative:
    every block is optimised at start 0). Returns latents (n, K, D) fp32 and the fraction of blocks exact."""
    c = model.c
    n, Bk, K, dev = blocks.shape[0], blocks.shape[1], c.k_lat, blocks.device
    pos = torch.cat([model.lat_pos(0, dev), torch.arange(Bk - 1, device=dev)])
    mem = torch.nn.Parameter(torch.randn(n, K, model.lm.get_input_embeddings().weight.shape[1], device=dev) * model.emb_rms)
    opt = torch.optim.Adam([mem], lr=lr)
    with torch.no_grad():
        tok_e = model.embed(blocks[:, :-1])
    exact = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.enable_grad(), model.lora(False):
        for _ in range(steps):
            logits = model.logits(model.hidden(torch.cat([mem.to(model.dtype), tok_e], 1), pos)[:, K - 1:])
            loss = F.cross_entropy(logits.flatten(0, 1), blocks.flatten())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            exact = (logits.argmax(-1) == blocks).all(1)
            if bool(exact.all()):
                break
    return mem.detach(), exact.float().mean().item()


@torch.no_grad()
def recall_eval(model, tokens, lc: LossCfg, tok=None, n_natural=16, n_probes=16, n_passkey=16, gain_nats=1.0,
                budgets=(0.02, 0.05, 0.1, 0.2), tokur=None, seed=0, latents="compressor", cram_steps=400):
    # tokur: one TokUR, or a dict {name: TokUR} to compare settings; signals are then tu<name>, au<name>, eu<name>
    """Scores RECALL DECISIONS position by position on the future region, against ground truth from actually
    expanding each block: gain_j(t) = CE_student(t) - CE_student-with-block-j-raw(t).
        when : which signal ranks positions where recall helps (best gain > gain_nats) first (AUROC), and how it
               tracks KL(teacher || student) (Spearman). Signals: student entropy, TokUR TU/AU/EU, trained div head.
        which: does a pointer pick the best block at helpful positions: trained pointer; attention mass on each
               block's latents (mean over layers and heads); attention lift (that mass / the block's mean mass
               over the example, which cancels attention sinks); lookahead (expand each block, keep the one that
               leaves the lowest entropy: label-free, costs one pass per block per recall); oracle.
        both : recall at the top-b fraction of positions by a signal, expanding the pointer's block: share of the
               total available gain captured (sum of positive best gains).
        gist : mean CE with latents vs with no memory per kind, and passkey answered from latents alone.
    latents="cram" replaces the compressor's latents with Phase-A latents optimised to regenerate each block
    (cram_blocks): do latents that demonstrably contain a block also work as soft memory?"""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    L_old, W, Fu, Bk, n = c.n_blocks * c.block, c.window, c.future, c.block, c.n_blocks
    L = sample_len(c)
    tokurs = tokur if isinstance(tokur, dict) else {"": tokur or TokUR(model)}
    unc = [f"{u}{name}" for name in tokurs for u in ("tu", "au", "eu")]
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = {k: [] for k in ("kind", "kl", "best_gain", "best_j", "entropy", "div", *unc,
                            "ptr_trained", "ptr_attn", "ptr_attn_lift", "ptr_lookahead", "ce_s", "ce_nomem")}
    gains_all, pk_latent, pk_oracle, cram_exact = [], [], [], []

    def examples():
        for _ in range(n_natural):
            yield "natural", make_batch(tokens, c, 1, False, dev)[0], None, None
        for _ in range(n_probes):
            yield "probe", make_batch(tokens, c, 1, True, dev)[0], None, None
        if tok is not None:
            for _ in range(n_passkey):
                x, j, n_ans = passkey_batch(tokens, c, tok, rng)
                yield "passkey", x.to(dev), j, n_ans

    for kind, x, pk_block, n_ans in examples():
        if latents == "none":                                          # control: window only; recall inserts raw
            lats = None
        elif latents == "cram":
            lat_all, ex = cram_blocks(model, x[0, :L_old].view(n, Bk), cram_steps)
            lats = [lat_all[i:i + 1] for i in range(n)]
            cram_exact.append(ex)
        else:
            lats = []
            for i in range(n):                                         # latents made once, at eviction time
                bs = i * Bk
                lats.append(model.compress([(l, j * Bk) for j, l in enumerate(lats)], x[:, bs:bs + Bk], bs,
                                           x[:, bs + Bk:bs + Bk + W], bs + Bk))
        y = x[0, L - Fu:]

        def parts_for(expand=None):
            if lats is None:
                return [] if expand is None else [("raw", x[:, expand * Bk:(expand + 1) * Bk], expand * Bk)]
            return [("raw", x[:, j * Bk:(j + 1) * Bk], j * Bk) if j == expand else ("lat", lats[j], j * Bk)
                    for j in range(n)]

        def read(expand=None, attentions=False):
            return model.student(parts_for(expand), x[:, L_old:L - 1], L_old, attentions=attentions)

        h, seg, off, att = read(attentions=True)
        s_idx = off + W - 1 + torch.arange(Fu, device=dev)
        logit_s = model.logits(h[:, s_idx])
        with model.lora(False):
            lt = F.log_softmax(model.logits(model.hidden(model.embed(x[:, :-1]), torch.arange(L - 1, device=dev))
                                            [:, L - Fu - 1:]), -1)
        ls = F.log_softmax(logit_s, -1)
        ce_s = -ls[0].gather(-1, y[:, None])[:, 0]
        gains, ents = [], []
        for j in range(n):
            hj, _, offj = read(expand=j)
            lj = F.log_softmax(model.logits(hj[:, offj + W - 1 + torch.arange(Fu, device=dev)]), -1)
            gains.append(ce_s + lj[0].gather(-1, y[:, None])[:, 0])
            ents.append(-(lj.exp() * lj).sum(-1)[0])
        gains = torch.stack(gains, 1)                                  # (Fu, n)
        for name, tk in tokurs.items():
            tu, au, eu = tk(lambda: model.logits(read()[0][:, s_idx]))
            rows[f"tu{name}"] += tu[0].tolist(); rows[f"au{name}"] += au[0].tolist(); rows[f"eu{name}"] += eu[0].tolist()
        att_mass = torch.stack([torch.stack([a[0][:, s_idx][:, :, seg == j].sum(-1) for j in range(n)], -1)
                                for a in att]).mean((0, 1))            # (Fu, n): layers x heads averaged
        if lats is None:                                               # nothing in context to point at
            att_mass = torch.full((Fu, n), float("nan"), device=dev)
            ptr = torch.full((Fu, n), float("nan"), device=dev)
        else:
            ptr = model.pointer_scores(model.block_summaries(h, seg, n), h[:, s_idx])[0]
        rows["kind"] += [kind] * Fu
        rows["kl"] += (lt.exp() * (lt - ls)).sum(-1)[0].tolist()
        rows["best_gain"] += gains.max(1).values.tolist()
        rows["best_j"] += gains.argmax(1).tolist()
        rows["entropy"] += (-(ls.exp() * ls).sum(-1))[0].tolist()
        rows["div"] += model.div_head(h[:, s_idx].float())[0, :, 0].tolist()
        valid = lats is not None                                       # -1 = pointer not available
        rows["ptr_trained"] += ptr.argmax(-1).tolist() if valid else [-1] * Fu
        rows["ptr_attn"] += att_mass.argmax(-1).tolist() if valid else [-1] * Fu
        rows["ptr_attn_lift"] += ((att_mass / att_mass.mean(0, keepdim=True).clamp_min(1e-12)).argmax(-1).tolist()
                                  if valid else [-1] * Fu)
        rows["ptr_lookahead"] += torch.stack(ents, 1).argmin(-1).tolist()
        h0, _, off0 = model.student([], x[:, L_old:L - 1], L_old)
        ce0 = F.cross_entropy(model.logits(h0[:, off0 + W - 1 + torch.arange(Fu, device=dev)])[0], y, reduction="none")
        rows["ce_s"] += ce_s.tolist(); rows["ce_nomem"] += ce0.tolist()
        if kind == "passkey":
            pk_latent.append(float((logit_s[0, :n_ans].argmax(-1) == y[:n_ans]).all()))
            hj, _, offj = read(expand=pk_block)
            pk_oracle.append(float((model.logits(hj[:, offj + W - 1 + torch.arange(n_ans, device=dev)])[0]
                                    .argmax(-1) == y[:n_ans]).all()))
        gains_all.append(gains.cpu())

    R = {k: np.array(v) for k, v in rows.items()}
    G = torch.cat(gains_all).numpy()                                   # (P, n) gain of expanding each block
    helpful = R["best_gain"] > gain_nats
    signals = ("entropy", "div", *unc)
    pointers = ("trained", "attn", "attn_lift", "lookahead")
    res = dict(positions=len(helpful), helpful_frac=float(helpful.mean()), n_blocks=n, signals=list(signals),
               pointers=list(pointers), budgets=list(budgets), latents=latents)
    for k in np.unique(R["kind"]):
        m = R["kind"] == k
        res[f"ce_latents_{k}"], res[f"ce_no_memory_{k}"] = float(R["ce_s"][m].mean()), float(R["ce_nomem"][m].mean())
    if pk_latent:
        res["passkey_from_latents"], res["passkey_block_expanded"] = float(np.mean(pk_latent)), float(np.mean(pk_oracle))
    if cram_exact:
        res["cram_blocks_exact"] = float(np.mean(cram_exact))
    for k in np.unique(R["kind"]):
        res[f"helpful_frac_{k}"] = float(helpful[R["kind"] == k].mean())
    for sname in signals:
        res[f"auroc_{sname}"] = auroc(R[sname], helpful)
        res[f"spearman_kl_{sname}"] = spearman(R[sname], R["kl"])
        for k in np.unique(R["kind"]):
            m = R["kind"] == k
            res[f"auroc_{sname}_{k}"] = auroc(R[sname][m], helpful[m])
    for p in pointers:
        res[f"ptr_{p}_acc_helpful"] = float((R["ptr_" + p][helpful] == R["best_j"][helpful]).mean()) \
            if helpful.any() and (R["ptr_" + p] >= 0).any() else float("nan")
    total = np.clip(R["best_gain"], 0, None).sum()
    res["ce_before_recall"] = float(R["ce_s"].mean())
    idx = np.arange(len(G))
    for sname in signals:
        for b in budgets:
            fire = R[sname] >= np.quantile(R[sname], 1 - b)
            for pname, choice in [("oracle", R["best_j"])] + [(p, R["ptr_" + p]) for p in pointers]:
                ok_ = fire & (choice >= 0)                             # unavailable pointer: no recall
                got = np.zeros(len(G)); got[ok_] = G[idx[ok_], choice[ok_]]
                res[f"capture_{sname}_{pname}_{b}"] = float(got.sum() / max(total, 1e-9))
                res[f"ce_after_{sname}_{pname}_{b}"] = float((R["ce_s"] - got).mean())   # comparable across modes
    model.train()
    return res, R


def print_recall_report(res):
    sig, ptrs, budgets = res["signals"], res["pointers"], res["budgets"]
    kinds = sorted(k[len("helpful_frac_"):] for k in res if k.startswith("helpful_frac_"))
    print(f"latents: {res.get('latents', 'compressor')}"
          + (f" (blocks regenerated exactly: {res['cram_blocks_exact']:.3f})" if "cram_blocks_exact" in res else ""))
    print("GIST: mean CE (nats) with latents vs no memory: " + " | ".join(
        f"{k} {res[f'ce_latents_{k}']:.3f} vs {res[f'ce_no_memory_{k}']:.3f}" for k in kinds))
    if "passkey_from_latents" in res:
        print(f"      passkey from latents alone {res['passkey_from_latents']:.3f}, "
              f"with its block expanded {res['passkey_block_expanded']:.3f}")
    print(f"positions {res['positions']}, recall helps (best block gain > threshold) at {res['helpful_frac']:.3f}"
          + "".join(f" | {k} {res['helpful_frac_' + k]:.3f}" for k in kinds))
    print(f"\nWHEN: AUROC for 'recall helps' (and Spearman with KL)")
    print(f"{'signal':>12} {'all':>7} " + " ".join(f"{k:>8}" for k in kinds) + f" {'rho_KL':>7}")
    for s_ in sig:
        print(f"{s_:>12} {res['auroc_' + s_]:>7.3f} " + " ".join(f"{res[f'auroc_{s_}_{k}']:>8.3f}" for k in kinds)
              + f" {res['spearman_kl_' + s_]:>7.3f}")
    print(f"\nWHICH: pointer picks the best block at helpful positions (chance {1 / res['n_blocks']:.3f}): "
          + " | ".join(f"{p} {res[f'ptr_{p}_acc_helpful']:.3f}" for p in ptrs))
    print(f"\nCE AFTER RECALL (nats, comparable across --latents modes; before recall {res['ce_before_recall']:.4f})")
    print(f"{'signal/pointer':>22} " + " ".join(f"{'b=' + str(b):>8}" for b in budgets))
    for s_ in ("entropy", "div"):
        for p in ("oracle", "lookahead"):
            print(f"{s_ + '/' + p:>22} " + " ".join(f"{res[f'ce_after_{s_}_{p}_{b}']:>8.4f}" for b in budgets))
    print("\nBOTH: share of total available gain captured, recalling at the top-b positions by signal")
    for p in ("oracle", *ptrs):
        print(f"  pointer = {p}")
        print(f"{'signal':>12} " + " ".join(f"{'b=' + str(b):>8}" for b in budgets))
        for s_ in sig:
            print(f"{s_:>12} " + " ".join(f"{res[f'capture_{s_}_{p}_{b}']:>8.3f}" for b in budgets))


# ============================================================================ end-to-end: needles in long documents

NAMES = ["Alice", "Bruno", "Chiara", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Ingrid", "Jonas", "Kaito", "Leila",
         "Marek", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Stefan", "Tamar", "Ulla", "Viktor", "Wanda", "Xavier",
         "Yusuf", "Zofia", "Anton", "Beatrix", "Cyril", "Dagny", "Emil", "Freya", "Gustav", "Hanna", "Ivo", "Juno"]


def needle_doc(tokens, c, tok, n_blocks, n_needles, rng):
    """n_blocks blocks of filler + a W-token recent window of filler. n_needles facts "The secret code of NAME is
    DDDDD." in distinct random blocks. Returns doc ids (L,) and [(name, code, block)]."""
    Bk, W = c.block, c.window
    L = n_blocks * Bk + W
    i = rng.randint(0, len(tokens) - L - 1)
    x = tokens[i:i + L].clone().long()
    needles = []
    for name, j in zip(rng.sample(NAMES, n_needles), rng.sample(range(n_blocks), n_needles)):
        code = str(rng.randint(10000, 99999))
        ids = tok.encode(f"\nThe secret code of {name} is {code}.\n")
        at = j * Bk + rng.randint(0, Bk - len(ids))
        x[at:at + len(ids)] = torch.tensor(ids)
        needles.append((name, code, j))
    return x, needles


@torch.no_grad()
def needle_eval(model, tokens, tok, n_blocks=16, n_docs=16, n_needles=4, taus=(0.0, 0.5, 1.0, 2.0, 3.0, float("inf")),
                prefilter=4, seed=0):
    """Long-document recall, end to end. The reader sees [latents of n_blocks evicted blocks | recent window |
    question] and must produce the 5-digit code (exact match, teacher-forced argmax = greedy decoding). Recall is
    decided at the first answer token: if the student's entropy there exceeds tau, a block is chosen and
    inserted raw. Modes:
        full      : the base model reading the whole raw document (long-context baseline)
        none      : window + question only
        latents   : latents, never recall (tau = inf)
        recall    : latents + entropy trigger + lookahead over all blocks (expand each, keep the lowest-entropy one)
        prefilter : as recall, but lookahead only over the trained pointer's top-k blocks (cheap at scale)
        nolat     : control without latents: entropy trigger + lookahead inserting raw blocks into window-only
        oracle    : latents + the needle's own block expanded
    Cost is counted as reader tokens processed per question (prefill), compressor passes excluded."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    Bk, W, K = c.block, c.window, c.k_lat
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rec = []                                                           # one dict per question
    for _ in range(n_docs):
        x, needles = needle_doc(tokens, c, tok, n_blocks, n_needles, rng)
        x = x.to(dev)[None]
        Ld = n_blocks * Bk
        lats = []
        for i in range(n_blocks):                                      # eviction-time compression; earlier latents
            bs = i * Bk                                                # condition it, as in training (<= n-1 of them)
            prev = [(lats[j], j * Bk) for j in range(max(0, i - (c.n_blocks - 1)), i)]
            lats.append(model.compress(prev, x[:, bs:bs + Bk], bs, x[:, bs + Bk:bs + Bk + W], bs + Bk))
        for name, code, jb in needles:
            q = tok.encode(f"\nWhat is the secret code of {name}? The secret code of {name} is")
            a = tok.encode(f" {code}")
            tail = torch.cat([x[0, Ld:], torch.tensor(q + a[:-1], device=dev)])[None]
            y = torch.tensor(a, device=dev)
            first = W + len(q) - 1                                     # tail index predicting the first answer token
            ans_idx = first + torch.arange(len(a), device=dev)

            def run(parts):
                h, seg, off = model.student(parts, tail, Ld)
                lg = model.logits(h[:, off + ans_idx])[0]
                ent = float(-(F.log_softmax(lg[0], -1).exp() * F.log_softmax(lg[0], -1)).sum())
                return bool((lg.argmax(-1) == y).all()), ent, h, seg, off, sum(p[1].shape[1] for p in parts) + tail.shape[1]

            lat_parts = lambda e=None: [("raw", x[:, j * Bk:(j + 1) * Bk], j * Bk) if j == e else ("lat", lats[j], j * Bk)
                                        for j in range(n_blocks)]
            raw_only = lambda e=None: [] if e is None else [("raw", x[:, e * Bk:(e + 1) * Bk], e * Bk)]
            r = dict(block=jb)
            with model.lora(False):                                    # full raw context, base model
                full = torch.cat([x[0], torch.tensor(q + a[:-1], device=dev)])[None]
                hf = model.hidden(model.embed(full), torch.arange(full.shape[1], device=dev))
                r["full"] = bool((model.logits(hf[:, Ld + ans_idx])[0].argmax(-1) == y).all())
                r["cost_full"] = full.shape[1]
            r["none"], r["ent_none"], *_, c_none = run([])
            r["latents"], r["ent_lat"], h, seg, off, c_lat = run(lat_parts())
            r["oracle"] = run(lat_parts(jb))[0]
            look = [run(lat_parts(j)) for j in range(n_blocks)]          # lookahead over all blocks
            jl = int(np.argmin([o[1] for o in look]))
            r["look_block"], r["recall_hit"], r["look_correct"] = jl, jl == jb, look[jl][0]
            ptr = model.pointer_scores(model.block_summaries(h, seg, n_blocks), h[:, off + first:off + first + 1])[0, 0]
            top = ptr.topk(min(prefilter, n_blocks)).indices.tolist()
            jp = top[int(np.argmin([look[j][1] for j in top]))]
            r["pre_block"], r["pre_hit"], r["pre_correct"] = jp, jp == jb, look[jp][0]
            r["ptr_in_top"] = jb in top
            look0 = [run(raw_only(j)) for j in range(n_blocks)]          # control: no latents
            j0 = int(np.argmin([o[1] for o in look0]))
            r["nolat_block"], r["nolat_hit"], r["nolat_correct"] = j0, j0 == jb, look0[j0][0]
            r["cost_none"], r["cost_lat"], r["cost_one_raw"] = c_none, c_lat, c_lat + Bk - K
            rec.append(r)

    m = lambda k: float(np.mean([r[k] for r in rec]))
    res = dict(n_blocks=n_blocks, questions=len(rec), full=m("full"), none=m("none"), latents=m("latents"),
               oracle=m("oracle"), lookahead_hit=m("recall_hit"), prefilter_hit=m("pre_hit"),
               pointer_topk_contains=m("ptr_in_top"), nolat_hit=m("nolat_hit"),
               cost_full=m("cost_full"), cost_latents=m("cost_lat"), cost_none=m("cost_none"))
    for t in taus:                                                     # trigger sweep: recall iff entropy > tau
        fire_l = [r["ent_lat"] > t for r in rec]
        fire_n = [r["ent_none"] > t for r in rec]
        res[f"recall@{t}"] = float(np.mean([r["look_correct"] if f else r["latents"] for r, f in zip(rec, fire_l)]))
        res[f"prefilter@{t}"] = float(np.mean([r["pre_correct"] if f else r["latents"] for r, f in zip(rec, fire_l)]))
        res[f"nolat@{t}"] = float(np.mean([r["nolat_correct"] if f else r["none"] for r, f in zip(rec, fire_n)]))
        res[f"rate_lat@{t}"], res[f"rate_none@{t}"] = float(np.mean(fire_l)), float(np.mean(fire_n))
    # reader tokens per question for one recall: lookahead tries every block, prefilter only k
    res["cost_recall_lookahead"] = res["cost_latents"] * (1 + n_blocks) + (Bk - K) * n_blocks
    res["cost_recall_prefilter"] = res["cost_latents"] * (1 + prefilter) + (Bk - K) * prefilter
    res["cost_nolat_lookahead"] = res["cost_none"] * (1 + n_blocks) + Bk * n_blocks
    model.train()
    return res, rec


def print_needle_report(res, taus=(0.0, 0.5, 1.0, 2.0, 3.0, float("inf"))):
    print(f"{res['questions']} questions, {res['n_blocks']} blocks per document (exact match of a 5-digit code)")
    print(f"  full raw document      {res['full']:.3f}   ({res['cost_full']:.0f} reader tokens)")
    print(f"  window only            {res['none']:.3f}   ({res['cost_none']:.0f})")
    print(f"  latents, no recall     {res['latents']:.3f}   ({res['cost_latents']:.0f})")
    print(f"  latents + oracle block {res['oracle']:.3f}")
    print(f"  block chosen: lookahead {res['lookahead_hit']:.3f} | pointer top-k + lookahead {res['prefilter_hit']:.3f} "
          f"(top-k contains it {res['pointer_topk_contains']:.3f}) | no-latents lookahead {res['nolat_hit']:.3f}")
    print(f"  cost of one recall (reader tokens): lookahead {res['cost_recall_lookahead']:.0f}, prefilter "
          f"{res['cost_recall_prefilter']:.0f}, no-latents lookahead {res['cost_nolat_lookahead']:.0f}")
    print(f"\n{'tau':>6} {'recall':>8} {'prefilt':>8} {'rate':>6} | {'nolat':>8} {'rate':>6}")
    for t in taus:
        print(f"{t:>6} {res[f'recall@{t}']:>8.3f} {res[f'prefilter@{t}']:>8.3f} {res[f'rate_lat@{t}']:>6.2f} | "
              f"{res[f'nolat@{t}']:>8.3f} {res[f'rate_none@{t}']:>6.2f}")


# ============================================================================ demo: look at what latents hold

DEMO_TEXT = ("The secret code of Alice is 48213. Bruno lives in Lisbon and works as a lighthouse keeper. "
             "The team meeting moved to Thursday at 3 pm in room 214, and Chiara will bring the blue folder.")


@torch.no_grad()
def demo(model, tok, text=DEMO_TEXT, prompts=("\nThe above repeated:\n",), n_new=48,
         modes=("raw", "none", "compressor", "compressor+cond", "cram"), cram_steps=600):
    """Compress `text` into latents, put `prompt` after them, and let the frozen base model continue greedily.
    The text is split into blocks of c.block tokens; each block becomes k_lat latents."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    ids = torch.tensor(tok.encode(text), device=dev)
    Bk, K = c.block, c.k_lat
    blocks = [ids[i:i + Bk] for i in range(0, len(ids), Bk)]
    T = len(ids)
    print(f"text: {T} tokens -> {len(blocks)} block(s) x {K} latents ({T / (len(blocks) * K):.1f} tokens per latent)")
    print(f"TEXT: {text!r}\n")

    def compress_all(cond):
        lats = []
        for i, b in enumerate(blocks):
            prev = [(lats[j], j * Bk) for j in range(max(0, i - (c.n_blocks - 1)), i)]
            lats.append(model.compress(prev, b[None], i * Bk, cond, T if cond is not None else None))
        return lats

    def cram_all():
        out, exact = [], []
        for b in blocks:
            m_, e = cram_blocks(model, b[None], cram_steps)
            out.append(m_); exact.append(e)
        return out, float(np.mean(exact))

    def generate(parts, prompt_ids):
        seq = prompt_ids.tolist()
        for _ in range(n_new):
            h, _, off = model.student(parts, torch.tensor(seq, device=dev)[None], T)
            nxt = int(model.logits(h[:, -1]).argmax(-1))
            seq.append(nxt)
            if nxt == tok.eos_token_id:
                break
        return tok.decode(seq[len(prompt_ids):])

    for prompt in prompts:
        p = torch.tensor(tok.encode(prompt), device=dev)
        print(f"PROMPT: {prompt!r}")
        for mode in modes:
            note = ""
            if mode == "raw":
                parts = [("raw", b[None], i * Bk) for i, b in enumerate(blocks)]
            elif mode == "none":
                parts = []
            elif mode == "compressor":
                parts = [("lat", l, i * Bk) for i, l in enumerate(compress_all(None))]
            elif mode == "compressor+cond":
                parts = [("lat", l, i * Bk) for i, l in enumerate(compress_all(p[None]))]
            elif mode == "cram":
                lats, ex = cram_all()
                parts = [("lat", l, i * Bk) for i, l in enumerate(lats)]
                note = f" (latents regenerate their block exactly: {ex:.0%})"
            print(f"  [{mode}]{note}\n    {generate(parts, p)!r}")
        print()
    model.train()


# ============================================================================ training

def load_tokens(path):
    return torch.from_numpy(np.load(path).astype(np.int64))


def train(args):
    dev = pick_device()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    c = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    lc = LossCfg(**{f.name: getattr(args, f.name) for f in fields(LossCfg)})
    tokens = load_tokens(args.tokens)
    n = int(0.95 * len(tokens))
    train_t, val_t = tokens[:n], tokens[n:]
    model = build(c, dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c.model)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"base {c.model} | trainable {sum(p.numel() for p in params) / 1e6:.1f}M | device {dev} | "
          f"teacher ctx {sample_len(c)} tok | student ctx {c.n_blocks * c.k_lat + c.window + c.future} | "
          f"compression {c.block // c.k_lat}x", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=(dev != "cpu"))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 and (step > 0 or args.eval_at_start) or step == args.steps:
            r = evaluate(model, val_t, lc, n_batches=args.eval_batches, bsz=args.batch, n_probes=args.eval_probes,
                         tok=tok, n_passkey=args.eval_passkey)
            print(f"== eval step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
            torch.save({"trainable": model.trainable_state(), "config": asdict(c), "loss_cfg": asdict(lc),
                        "step": step}, args.out)
            t0 = time.time()
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        probe = random.random() < args.probe_frac
        expand_idx = random.randrange(c.n_blocks) if random.random() < args.expand_prob else None
        recond = random.random() < args.recond_prob
        drop_window = random.random() < args.drop_window_prob
        x, src = make_batch(train_t, c, args.batch, probe, dev)
        o = compute(model, x, src, lc, expand_idx=expand_idx, recond=recond, drop_window=drop_window)
        o["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = (time.time() - t0) / (args.log_every if step else 1) * 1000
            t0 = time.time()
            mem = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev == "cuda" else ""
            keys = [k for k in ("loss", "ce_t", "ce_s", "kl", "div", "ptr") if k in o]
            print(f"step {step} | " + " ".join(f"{k} {o[k].item():.4f}" for k in keys) +
                  f" | gn {gn:.2f} | lr {lr_at(step):.2e} | {'probe ' if probe else ''}"
                  f"{'exp' + str(expand_idx) + ' ' if expand_idx is not None else ''}{'recond ' if recond else ''}"
                  f"{'nowin ' if drop_window else ''}| {dt:.0f} ms/step{mem}", flush=True)


def load(path):
    dev = pick_device()
    ck = torch.load(path, map_location=dev)
    c = Config(**{"reader_lora": True, **ck["config"]})   # checkpoints before reader_lora trained the reader
    model = build(c, dev)
    missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
    assert not unexpected, unexpected
    return model.eval(), LossCfg(**ck["loss_cfg"]), c


def prep(args):
    """Tokenise a raw text/bytes file with the base model's tokenizer into a uint32 .npy."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    raw = open(args.data, "rb").read().decode("utf-8", errors="replace")
    chunks, step = [], 1 << 20
    for i in range(0, len(raw), step):
        chunks.append(np.array(tok.encode(raw[i:i + step]), dtype=np.uint32))
    arr = np.concatenate(chunks)
    np.save(args.out, arr)
    print(f"{len(raw)} chars -> {len(arr)} tokens ({len(raw) / len(arr):.2f} chars/token) -> {args.out}")


# ============================================================================ selftest

def selftest():
    """Tiny random Qwen3 (no download). Checks the teacher stays the base model, causality, information flow
    through latents, finite losses and gradients in every batch mode, overfitting, evaluation."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0); random.seed(0)
    dev = pick_device()
    qc = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                     tie_word_embeddings=True)
    lm = Qwen3ForCausalLM(qc).float()
    lm.config._attn_implementation = "sdpa"
    c = Config(model="tiny", block=32, k_lat=4, n_blocks=3, window=32, future=32, lora_r=4, ptr_dim=16)
    lc = LossCfg()
    model = build(c, dev, lm)
    ok = lambda s: print(f"  [ok] {s}")
    tokens = torch.randint(0, 512, (20000,))
    L_old, L = c.n_blocks * c.block, sample_len(c)

    x, _ = make_batch(tokens, c, 2, False, dev)
    with torch.no_grad():
        base_logits = model.logits(model.hidden(model.embed(x), torch.arange(L, device=dev)))

    # 1. student causality and information flow
    def stud(xx, expand=None):
        lats = []
        for i in range(c.n_blocks):
            bs = i * c.block
            lats.append(model.compress([(l, j * c.block) for j, l in enumerate(lats)], xx[:, bs:bs + c.block], bs,
                                       xx[:, bs + c.block:bs + c.block + c.window], bs + c.block))
        parts = [("raw", xx[:, j * c.block:(j + 1) * c.block], j * c.block) if j == expand else
                 ("lat", lats[j], j * c.block) for j in range(c.n_blocks)]
        return model.student(parts, xx[:, L_old:], L_old)

    with torch.no_grad():
        h1, seg, off = stud(x)
        assert h1.shape[1] == c.n_blocks * c.k_lat + c.window + c.future
        j = c.window + 5
        x3 = x.clone(); x3[:, L_old + j] = (x3[:, L_old + j] + 1) % 512
        assert torch.allclose(h1[:, :off + j], stud(x3)[0][:, :off + j], atol=1e-5)
        ok("student is causal w.r.t. the future region")
        x4 = x.clone(); x4[:, 3] = (x4[:, 3] + 1) % 512
        assert not torch.allclose(h1[:, off + c.window:], stud(x4)[0][:, off + c.window:], atol=1e-6)
        ok("old-block content reaches the future region through latents")
        h5, seg5, _ = stud(x, expand=1)
        assert h5.shape[1] == h1.shape[1] + c.block - c.k_lat and (seg5 == 1).sum() == c.block
        ok("expanded layout has the right shape")
        lat = model.compress([], x[:, :c.block], 0, x[:, c.block:c.block + c.window], c.block)
        r = float(lat.pow(2).mean(-1).sqrt().mean() / model.emb_rms)
        assert 0.5 < r < 2.0, r
        ok(f"latents are at token-embedding scale (RMS ratio {r:.2f})")

    # 2. every batch mode: finite losses, gradients reach all trainable parts
    for probe in (False, True):
        for expand in (None, 1):
            for recond in (False, True):
                for nowin in (False, True):
                    xb, src = make_batch(tokens, c, 2, probe, dev)
                    o = compute(model, xb, src, lc, expand_idx=expand, recond=recond, drop_window=nowin)
                    assert torch.isfinite(o["loss"]), (probe, expand, recond, nowin)
                    o["loss"].backward()
    for name in ("beacons", "lat_type"):
        assert getattr(model, name).grad.abs().sum() > 0, name
    for mod in ("adapter", "div_head", "ptr_q", "ptr_k", "lat_out_norm"):
        assert sum(p.grad.abs().sum().item() for p in getattr(model, mod).parameters()) > 0, mod
    assert all(lo.B.grad is not None and lo.B.grad.abs().sum() > 0 for lo in model.loras)
    assert all(not p.requires_grad for p in model.lm.parameters() if not any(p is q for lo in model.loras for q in (lo.A, lo.B)))
    ok("losses finite in all 16 batch modes; gradients reach LoRA, compressor and heads; base frozen")
    model.zero_grad(set_to_none=True)
    xb, src = make_batch(tokens, c, 2, True, dev)
    o = compute(model, xb, src, lc)
    (o["ptr"] + o["div"]).backward()
    assert model.beacons.grad is None or model.beacons.grad.abs().sum() == 0
    assert all(lo.B.grad is None or lo.B.grad.abs().sum() == 0 for lo in model.loras)
    assert model.ptr_q.weight.grad.abs().sum() > 0 and model.div_head.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    ok("recall heads learn from detached states: pointer/div losses do not reach the compressor (head_grad off)")

    # 3. overfit a fixed probe batch; the teacher (LoRA off) must still be the untouched base model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    xb, src = make_batch(tokens, c, 4, True, dev)
    first = None
    for _ in range(150):
        o = compute(model, xb, src, lc)
        first = first if first is not None else o["loss"].item()
        opt.zero_grad(); o["loss"].backward(); opt.step()
    # weak bar: the (random, tiny) base is frozen and, with reader_lora off, only the compressor learns; on random
    # tokens a random frozen reader can use latents only a little (byte model: 0.7; reader LoRA on: ~0.76)
    assert o["loss"].item() < 0.95 * first, (first, o["loss"].item())
    ok(f"overfits a fixed batch: loss {first:.3f} -> {o['loss'].item():.3f}")
    with torch.no_grad(), model.lora(False):
        after = model.logits(model.hidden(model.embed(x), torch.arange(L, device=dev)))
    assert torch.allclose(base_logits, after, atol=1e-5)
    ok("teacher (LoRA off) is unchanged by training")
    with torch.no_grad():
        hr, _, _ = model.student([], x[:, L_old:], L_old)
        with model.lora(False):
            hb = model.hidden(model.embed(x[:, L_old:]), torch.arange(L_old, L, device=dev))
    assert torch.allclose(hr, hb, atol=1e-5)
    ok("reader is the frozen base model (reader_lora off)")

    # 4. recall decisions: TokUR sanity, eager attention matches sdpa, the eval runs
    with torch.no_grad():
        parts = [("lat", model.compress([], x[:, :c.block], 0, x[:, c.block:c.block + c.window], c.block), 0)]
        hs, _, _ = model.student(parts, x[:, L_old:], L_old)
        he, _, _, att = model.student(parts, x[:, L_old:], L_old, attentions=True)
        assert torch.allclose(hs, he, atol=1e-4) and len(att) == qc.num_hidden_layers
        assert torch.allclose(att[0].sum(-1), torch.ones_like(att[0].sum(-1)), atol=1e-4)
        fn = lambda: model.logits(model.student(parts, x[:, L_old:], L_old)[0])
        tu, au, eu = TokUR(model, r=4, sigma=0.0, M=2)(fn)
        assert eu.abs().max() < 1e-4 and torch.allclose(tu, au, atol=1e-4)
        tu, au, eu = TokUR(model, r=4, sigma=0.5, M=3)(fn)
        assert eu.min() > -1e-4 and eu.max() > 1e-4 and all(lo.perturb is None for lo in model.loras)
    ok("TokUR: no perturbation -> EU 0; perturbation -> EU >= 0, removed afterwards; eager attention == sdpa")
    # gain threshold 0: on a random tiny model no block gains a full nat, which would leave every AUROC undefined
    res, R = recall_eval(model, tokens, lc, n_natural=2, n_probes=2, gain_nats=0.0, tokur=TokUR(model, r=4, sigma=0.1))
    assert res["positions"] == 4 * c.future and all(np.isfinite(R[k]).all() for k in ("kl", "eu", "best_gain"))
    assert 0 < res["helpful_frac"] < 1, res["helpful_frac"]
    assert all(np.isfinite(res[k]) for k in ("auroc_entropy", "auroc_eu", "auroc_div", "ptr_trained_acc_helpful",
                                              "ptr_attn_acc_helpful", "ptr_attn_lift_acc_helpful",
                                              "ptr_lookahead_acc_helpful", "capture_eu_oracle_0.1",
                                              "capture_entropy_lookahead_0.1"))
    assert res["capture_entropy_oracle_0.2"] >= res["capture_entropy_trained_0.2"] - 1e-9   # oracle is an upper bound
    ok(f"recall_eval runs: helpful {res['helpful_frac']:.3f}, AUROC entropy {res['auroc_entropy']:.3f}, "
       f"EU {res['auroc_eu']:.3f}, pointer trained/attn {res['ptr_trained_acc_helpful']:.2f}/"
       f"{res['ptr_attn_acc_helpful']:.2f}")
    res_n, _ = recall_eval(model, tokens, lc, n_natural=1, n_probes=1, gain_nats=0.0, tokur=TokUR(model, r=4),
                           latents="none")
    assert np.isnan(res_n["ptr_trained_acc_helpful"]) or res_n["ptr_trained_acc_helpful"] == 0
    assert np.isfinite(res_n["ce_after_entropy_lookahead_0.1"]) and abs(res_n["ce_latents_natural"] -
                                                                          res_n["ce_no_memory_natural"]) < 1e-5
    ok("recall_eval --latents none (control) runs: no latents, recall inserts raw blocks, CE-after comparable")
    res_c, _ = recall_eval(model, tokens, lc, n_natural=1, n_probes=1, gain_nats=0.0, tokur=TokUR(model, r=4),
                           latents="cram", cram_steps=30)
    assert np.isfinite(res_c["ce_latents_natural"]) and 0 <= res_c["cram_blocks_exact"] <= 1
    ok(f"recall_eval with Phase-A (cram) latents runs: blocks exact {res_c['cram_blocks_exact']:.2f} after 30 steps")
    assert abs(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    ok("auroc matches a hand-computed case")

    class FakeTok:                                                     # word-level ids for the tiny vocab, so a
        encode = staticmethod(lambda t: [sum(map(ord, w)) % 512 for w in t.split()])   # needle fits in a 32-token block
    res_e, rec_e = needle_eval(model, tokens, FakeTok, n_blocks=5, n_docs=2, n_needles=2, prefilter=2)
    assert res_e["questions"] == 4 and all(0 <= res_e[k] <= 1 for k in ("full", "none", "latents", "oracle"))
    assert res_e["cost_full"] > res_e["cost_latents"] > res_e["cost_none"]
    assert all(r["look_block"] in range(5) and r["pre_block"] in range(5) for r in rec_e)
    print_needle_report(res_e)
    ok("needle_eval runs end to end (long documents, eviction-time compression, recall modes, costs)")

    class DemoTok(FakeTok):
        eos_token_id = -1
        decode = staticmethod(lambda ids: " ".join(map(str, ids)))
    demo(model, DemoTok, "one two three four five six seven eight", ("repeat it",), n_new=4, cram_steps=5)
    ok("demo runs in every mode")

    # 5. evaluation runs (passkey needs a tokenizer; skipped here)
    r = evaluate(model, tokens, lc, n_batches=2, bsz=2, n_probes=3)
    assert all(np.isfinite(v) for v in r.values()), r
    ok("evaluate runs: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items()))
    print("SELFTEST PASSED")


# ============================================================================ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=Config.model)

    t = sub.add_parser("train")
    t.add_argument("--tokens", required=True)
    t.add_argument("--out", default="rolling_qwen.pt")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--probe_frac", type=float, default=0.5)
    t.add_argument("--expand_prob", type=float, default=0.3)
    t.add_argument("--recond_prob", type=float, default=0.2)
    t.add_argument("--drop_window_prob", type=float, default=0.25)
    t.add_argument("--eval_every", type=int, default=500)
    t.add_argument("--eval_at_start", action="store_true", help="eval before training (untrained compressor)")
    t.add_argument("--eval_batches", type=int, default=10)
    t.add_argument("--eval_probes", type=int, default=32)
    t.add_argument("--eval_passkey", type=int, default=32)
    t.add_argument("--log_every", type=int, default=25)
    t.add_argument("--seed", type=int, default=0)
    for f in fields(Config) + fields(LossCfg):
        if f.type in (bool, "bool"):
            t.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            t.add_argument(f"--{f.name}", type=typ, default=f.default)

    rc = sub.add_parser("recall_eval", help="score recall decisions (when: TokUR/entropy/div head; which: pointers)")
    rc.add_argument("--ckpt", required=True)
    rc.add_argument("--tokens", required=True)
    rc.add_argument("--natural", type=int, default=16)
    rc.add_argument("--probes", type=int, default=16)
    rc.add_argument("--passkey", type=int, default=16)
    rc.add_argument("--gain_nats", type=float, default=1.0, help="recall 'helps' if expanding a block gains this")
    rc.add_argument("--rank", type=int, default=8)
    rc.add_argument("--sigma", type=float, nargs="+", default=[0.1], help="one TokUR per sigma (a sweep)")
    rc.add_argument("--M", type=int, default=2)
    rc.add_argument("--latents", choices=("compressor", "cram", "none"), default="compressor",
                    help="cram: Phase-A latents optimised to regenerate each block, instead of the compressor's; "
                         "none: control with no latents (window only, recall inserts the raw block)")
    rc.add_argument("--cram_steps", type=int, default=600)
    rc.add_argument("--out", default=None, help="save the summary (.json) and per-position arrays (.npz)")

    ne = sub.add_parser("needle_eval", help="end-to-end: needles in long documents, recall vs baselines")
    ne.add_argument("--ckpt", required=True)
    ne.add_argument("--tokens", required=True)
    ne.add_argument("--blocks", type=int, nargs="+", default=[16, 64])
    ne.add_argument("--docs", type=int, default=16)
    ne.add_argument("--needles", type=int, default=4)
    ne.add_argument("--prefilter", type=int, default=4)
    ne.add_argument("--out", default=None)

    dm = sub.add_parser("demo", help="compress a text into latents, add a prompt, print what the model generates")
    dm.add_argument("--ckpt", required=True)
    dm.add_argument("--text", default=DEMO_TEXT)
    dm.add_argument("--prompt", nargs="+", default=["\nThe above repeated:\n",
                                                   "\nQuestion: What is the secret code of Alice?\nAnswer:"])
    dm.add_argument("--modes", nargs="+", default=["raw", "none", "compressor", "compressor+cond", "cram"])
    dm.add_argument("--n", type=int, default=48)
    dm.add_argument("--cram_steps", type=int, default=600)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--tokens", required=True)
    e.add_argument("--batches", type=int, default=20)
    e.add_argument("--probes", type=int, default=64)
    e.add_argument("--passkey", type=int, default=64)

    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "prep":
        prep(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "demo":
        from transformers import AutoTokenizer
        model, lc, c = load(args.ckpt)
        tok = AutoTokenizer.from_pretrained(c.model)
        prompts = [p.encode().decode("unicode_escape") for p in args.prompt]   # allow \n on the command line
        demo(model, tok, args.text, prompts, args.n, args.modes, args.cram_steps)
    elif args.cmd == "needle_eval":
        import json
        from transformers import AutoTokenizer
        model, lc, c = load(args.ckpt)
        tokens = load_tokens(args.tokens)
        tok = AutoTokenizer.from_pretrained(c.model)
        allres = []
        for nb in args.blocks:
            res, rec = needle_eval(model, tokens[int(0.95 * len(tokens)):], tok, n_blocks=nb, n_docs=args.docs,
                                   n_needles=args.needles, prefilter=args.prefilter)
            print_needle_report(res)
            print()
            allres.append(res)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(allres, f, indent=1)
    elif args.cmd == "recall_eval":
        import json
        from transformers import AutoTokenizer
        model, lc, c = load(args.ckpt)
        tokens = load_tokens(args.tokens)
        res, R = recall_eval(model, tokens[int(0.95 * len(tokens)):], lc, tok=AutoTokenizer.from_pretrained(c.model),
                             n_natural=args.natural, n_probes=args.probes, n_passkey=args.passkey,
                             gain_nats=args.gain_nats,
                             tokur={("" if len(args.sigma) == 1 else f"@{sg}"): TokUR(model, args.rank, sg, args.M)
                                    for sg in args.sigma},
                             latents=args.latents, cram_steps=args.cram_steps)
        print_recall_report(res)
        if args.out:
            base = os.path.splitext(args.out)[0]
            with open(base + ".json", "w") as f:
                json.dump(dict(res, ckpt=args.ckpt, rank=args.rank, sigma=args.sigma, M=args.M), f, indent=1)
            np.savez(base + ".npz", **R)
    else:
        from transformers import AutoTokenizer
        model, lc, c = load(args.ckpt)
        tokens = load_tokens(args.tokens)
        r = evaluate(model, tokens[int(0.95 * len(tokens)):], lc, n_batches=args.batches, n_probes=args.probes,
                     tok=AutoTokenizer.from_pretrained(c.model), n_passkey=args.passkey)
        for k, v in r.items():
            print(f"{k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
