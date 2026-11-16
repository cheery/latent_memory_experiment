#!/usr/bin/env python3
"""
rolling_latent_lm.py

Byte-level LM that keeps only a short raw window in context and continuously compresses older text
into latent vectors, trained so that its behavior with compressed context matches its own behavior
with the full raw context. Raw text is never thrown away: compressed blocks can be re-expanded
("recalled") when a learned head predicts the latents are not enough.

Three tiers at inference:
    raw store (all bytes, outside the context)  ->  latents in context (K vectors per block)
                                                ->  recent raw window in context

Training (one shared transformer plays every role):
    teacher : the model on the full raw sequence                [old blocks | recent | future]
    student : the model on                                       [latents    | recent | future]
              (optionally with one block left raw = "expanded")
    compressor: the same model reads [previous latents | block | conditioning window | K beacons];
              beacon outputs -> adapter -> latents. Conditioning window = the bytes right after the
              block (what a streaming model sees at eviction time) or, for "recond" batches, the
              current recent window (what re-compression at inference uses).

Losses:
    lm    : next-byte CE of the teacher on the whole raw sequence (keeps the base LM trained)
    ce_s  : next-byte CE of the student on the future region
    kl    : KL(teacher || student) on the future region, teacher detached  (behavior preservation)
    div   : regression head predicting log1p(KL) per position   (does the model know what it lost?)
    ptr   : on copy probes, a pointer head predicts which old block holds the needed bytes (recall)

Copy probes: some batches replace the future region with [PROBE] + a verbatim span from the old
region. The teacher can copy it from raw context; the student must preserve it in latents or recall.

Commands:
    python rolling_latent_lm.py selftest
    python rolling_latent_lm.py train --data input.txt --out model.pt
    python rolling_latent_lm.py eval --ckpt model.pt --data input.txt
    python rolling_latent_lm.py generate --ckpt model.pt --prompt "..." --n 1000

STATUS: selftest passes on CPU and XPU. See rolling_latent_lm.md.
"""
import argparse
import math
import random
import time
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROBE = 256        # special token that starts a copy probe
VOCAB = 260        # 256 bytes + PROBE + 3 spare
IGNORE = -100


def pick_device():
    """CUDA, else XPU, else CPU; override with RLLM_DEVICE=cpu (or xpu/cuda)."""
    import os
    d = os.environ.get("RLLM_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    return "xpu" if torch.xpu.is_available() else "cpu"


@dataclass
class Config:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    block: int = 256       # bytes per compressible block
    k_lat: int = 16        # latents per block -> compression ratio block/k_lat
    n_blocks: int = 6      # old blocks per training sample
    window: int = 256      # recent raw window kept in context
    future: int = 256      # scored region after the window; keep == block (see HANDOFF.md)
    ptr_dim: int = 128
    rope_base: float = 10000.0
    lat_norm: bool = True  # RMS-normalise latents to token-embedding scale (False = original, latents blew up)


@dataclass
class LossCfg:
    w_ce: float = 1.0
    w_kl: float = 1.0
    w_div: float = 0.1
    w_ptr: float = 0.5
    distill_start: int = 1000   # steps of plain LM training before the student is trained
    kl_ramp: int = 1000         # linear ramp of the KL weight after distill_start
    detach_prev: bool = False   # detach earlier latents when they condition later compression
    probe_cue: int = 16         # first span bytes shown before copy accuracy / pointer are scored


# ============================================================================ model

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, pos, base):
    """x: (B, H, T, dh), pos: (T,) integer positions. Relative: shifting all positions changes nothing."""
    dh = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d_model
        self.h, self.base = c.n_heads, c.rope_base
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, pos, cache=None):
        """cache: None, or a dict this layer fills with its keys/values. An empty dict = prefill
        (causal over x); a filled one = decode, x must be a single new token appended after it."""
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos, self.base), rope(k, pos, self.base)
        if cache is not None and "k" in cache:
            assert T == 1, "decode with a KV cache takes one token at a time"
            k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            y = F.scaled_dot_product_attention(q, k, v)       # one query sees every cached key
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class RollingLatentLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.block % c.k_lat == 0, "block must be divisible by k_lat"
        assert c.future - 1 <= c.n_blocks * c.block, "probe span must fit in the old region"
        self.c = c
        D = c.d_model
        self.emb = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)
        # compression
        self.beacons = nn.Parameter(torch.randn(c.k_lat, D) * 0.02)
        self.adapter = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.lat_type = nn.Parameter(torch.zeros(D))
        self.lat_out_norm = RMSNorm(D) if c.lat_norm else None
        # recall
        self.div_head = nn.Linear(D, 1)
        self.ptr_q = nn.Linear(D, c.ptr_dim, bias=False)
        self.ptr_k = nn.Linear(D, c.ptr_dim, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def run(self, x, pos, caches=None):
        """caches: None, or a list with one dict per layer (see Layer.forward)."""
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, None if caches is None else caches[i])
        return self.norm(x)

    def lat_pos(self, start, device):
        """K latents of a block sit at the ends of its K sub-spans, so relative distances match raw text."""
        s = self.c.block // self.c.k_lat
        return start + torch.arange(self.c.k_lat, device=device) * s + (s - 1)

    def _ar(self, start, n, device):
        return start + torch.arange(n, device=device)

    def compress(self, prev, block_bytes, block_start, cond_bytes, cond_start):
        """prev: list of (latents (B,K,D), block_start). Returns latents (B,K,D) for this block."""
        dev, B, K = block_bytes.device, block_bytes.shape[0], self.c.k_lat
        dt = self.emb.weight.dtype
        xs, ps = [], []
        for lat, st in prev:
            xs.append(lat.to(dt))
            ps.append(self.lat_pos(st, dev))
        xs.append(self.emb(block_bytes))
        ps.append(self._ar(block_start, block_bytes.shape[1], dev))
        if cond_bytes is not None and cond_bytes.shape[1] > 0:
            xs.append(self.emb(cond_bytes))
            ps.append(self._ar(cond_start, cond_bytes.shape[1], dev))
        pos = torch.cat(ps)
        bpos = torch.full((K,), int(pos.max()) + 1, device=dev, dtype=torch.long)
        xs.append(self.beacons.to(dt).unsqueeze(0).expand(B, -1, -1))
        h = self.run(torch.cat(xs, 1), torch.cat([pos, bpos]))
        lat = self.adapter(h[:, -K:])
        if self.lat_out_norm is not None:
            # unnormalised latents grew to ~500x token-embedding RMS and became attention sinks with no content
            scale = self.emb.weight[:256].detach().pow(2).mean().sqrt()
            lat = self.lat_out_norm(lat) * scale
        return lat + self.lat_type

    def student(self, parts, tail, tail_start, caches=None):
        """parts: list of ('lat', (B,K,D), start) or ('raw', (B,L) long, start), in text order.
        Returns hidden (B,T,D), seg (T,) = part index per position (-1 for tail), offset of tail."""
        dev = tail.device
        dt = self.emb.weight.dtype
        xs, ps, seg = [], [], []
        for j, (kind, t, st) in enumerate(parts):
            if kind == "lat":
                xs.append(t.to(dt))
                p = self.lat_pos(st, dev)
            else:
                xs.append(self.emb(t))
                p = self._ar(st, t.shape[1], dev)
            ps.append(p)
            seg.append(torch.full_like(p, j))
        off = sum(x.shape[1] for x in xs)
        xs.append(self.emb(tail))
        p = self._ar(tail_start, tail.shape[1], dev)
        ps.append(p)
        seg.append(torch.full_like(p, -1))
        h = self.run(torch.cat(xs, 1), torch.cat(ps), caches)
        return h, torch.cat(seg), off

    def pointer_logits(self, h, seg, n_parts, query_idx):
        """Score each context part (compressed or raw block) for each query position. -> (B, Q, n_parts)"""
        return self.pointer_scores(self.block_summaries(h, seg, n_parts), h[:, query_idx])

    def block_summaries(self, h, seg, n_parts):
        """Mean hidden state of each context part -> (B, n_parts, D). Parts precede the tail and the model
        is causal, so these do not change while bytes are appended (the KV-cached decoder keeps them)."""
        return torch.stack([h[:, seg == j].mean(1) for j in range(n_parts)], 1)

    def pointer_scores(self, summaries, hq):
        """summaries (B, n_parts, D), hq (B, Q, D) -> (B, Q, n_parts)"""
        q, k = self.ptr_q(hq), self.ptr_k(summaries)
        return q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])


# ============================================================================ data and losses

def make_batch(data, c: Config, bsz, probe, device):
    """Returns x (B, L) long and src (B,) start of the copied span (or None)."""
    L_old = c.n_blocks * c.block
    L = L_old + c.window + c.future
    ix = torch.randint(0, len(data) - L - 1, (bsz,))
    x = torch.stack([data[i:i + L] for i in ix.tolist()]).long()
    src = None
    if probe:
        span = c.future - 1
        fut0 = L_old + c.window
        src = torch.zeros(bsz, dtype=torch.long)
        for b in range(bsz):
            s = random.randint(0, L_old - span)
            x[b, fut0] = PROBE
            x[b, fut0 + 1:] = x[b, s:s + span]
            src[b] = s
        src = src.to(device)
    return x.to(device), src


def compute(model, x, src, lc: LossCfg, expand_idx=None, recond=False, step=10**9, student=True, memory=True,
            drop_window=False):
    """Shared by training, eval and selftest. Returns a dict of losses and intermediates.
    memory=False: control with no latents at all (student sees only recent window + future); eval only.
    drop_window: the student sees [latents | future] with no recent window, so the future can only be
    predicted through the latents (training pressure against ignoring them)."""
    assert memory or not drop_window, "nothing would precede the first future byte"
    c = model.c
    out = {}
    B_, L = x.shape
    L_old = c.n_blocks * c.block
    W, Fu, Bk = c.window, c.future, c.block
    dev = x.device

    y = x[:, 1:].clone()
    y[y == PROBE] = IGNORE
    h_t = model.run(model.emb(x), torch.arange(L, device=dev))
    logit_t = model.head(h_t[:, :-1]).float()                       # predicts x[:, 1:]
    out["lm"] = F.cross_entropy(logit_t.reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGNORE)
    out["loss"] = out["lm"]
    if not student or step < lc.distill_start:
        return out

    # --- compress old blocks, sequentially, each conditioned on earlier latents
    lats = []
    for i in range(c.n_blocks if memory else 0):
        bs = i * Bk
        cs = L_old if recond else bs + Bk
        prev = [((l.detach() if lc.detach_prev else l), j * Bk) for j, l in enumerate(lats)]
        lats.append(model.compress(prev, x[:, bs:bs + Bk], bs, x[:, cs:cs + W], cs))

    parts = [("raw", x[:, j * Bk:(j + 1) * Bk], j * Bk) if j == expand_idx else ("lat", lats[j], j * Bk)
             for j in range(c.n_blocks)] if memory else []
    Wk = 0 if drop_window else W                                      # recent-window bytes the student sees
    tail = x[:, L_old + W - Wk:]                                      # recent window + future
    h_s, seg, off = model.student(parts, tail, L_old + W - Wk)

    # --- score ONLY the future region. Latents may encode the recent window (conditioning), so
    #     predicting inside the window would leak; predicting after it is what streaming does.
    s_idx = off + Wk - 1 + torch.arange(Fu, device=dev)              # student positions predicting future
    hs_f = h_s[:, s_idx]
    logit_s = model.head(hs_f).float()
    logit_tf = logit_t[:, L - Fu - 1:L - 1]                           # teacher positions predicting future
    y_f = y[:, L - Fu - 1:]                                           # == x[:, L-Fu:] with PROBE masked
    mask = (y_f != IGNORE).float()

    out["ce_s"] = F.cross_entropy(logit_s.reshape(-1, VOCAB), y_f.reshape(-1), ignore_index=IGNORE)
    out["ce_t_future"] = F.cross_entropy(logit_tf.reshape(-1, VOCAB), y_f.reshape(-1), ignore_index=IGNORE)
    lt = F.log_softmax(logit_tf.detach(), -1)
    ls = F.log_softmax(logit_s, -1)
    kl_pos = (lt.exp() * (lt - ls)).sum(-1)                           # KL(teacher || student) per position
    out["kl"] = (kl_pos * mask).sum() / mask.sum()
    div_pred = model.div_head(hs_f).squeeze(-1).float()
    out["div"] = (((div_pred - torch.log1p(kl_pos.detach().clamp_min(0))) ** 2) * mask).sum() / mask.sum()

    ramp = min(1.0, max(0.0, (step - lc.distill_start) / max(1, lc.kl_ramp)))
    out["loss"] = out["lm"] + lc.w_ce * out["ce_s"] + lc.w_kl * ramp * out["kl"] + lc.w_div * out["div"]

    if src is not None and memory:
        # span byte m sits at tail index W+1+m and is predicted from tail index W+m
        span = Fu - 1
        m = torch.arange(lc.probe_cue, span, device=dev)
        q_idx = off + Wk + m
        plog = model.pointer_logits(h_s, seg, c.n_blocks, q_idx).float()   # (B, Q, n_blocks)
        tgt = (src[:, None] + m[None, :]) // Bk
        out["ptr"] = F.cross_entropy(plog.reshape(-1, c.n_blocks), tgt.reshape(-1))
        out["loss"] = out["loss"] + lc.w_ptr * out["ptr"]
        out["ptr_logits"], out["ptr_tgt"] = plog, tgt

    out.update(logit_s=logit_s, logit_tf=logit_tf, y_f=y_f, mask=mask, kl_pos=kl_pos, div_pred=div_pred)
    return out


# ============================================================================ evaluation

@torch.no_grad()
def no_window_ce(model, x):
    """CE on future bytes 1.. with no recent window: (with latents, with nothing). Same targets for both."""
    c = model.c
    L_old, W = c.n_blocks * c.block, c.window
    lats = []
    for i in range(c.n_blocks):
        bs = i * c.block
        lats.append(model.compress([(l, j * c.block) for j, l in enumerate(lats)],
                                   x[:, bs:bs + c.block], bs, x[:, bs + c.block:bs + c.block + W], bs + c.block))
    y = x[:, L_old + W + 1:]
    res = []
    for parts in ([("lat", lats[j], j * c.block) for j in range(c.n_blocks)], []):
        h, _, off = model.student(parts, x[:, L_old + W:], L_old + W)
        res.append(F.cross_entropy(model.head(h[:, off:-1]).float().reshape(-1, VOCAB), y.reshape(-1)).item())
    return res

@torch.no_grad()
def evaluate(model, data, lc: LossCfg, n_batches=10, bsz=8, n_probes=32):
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    ln2 = math.log(2)
    res = {}

    # natural text: does the student behave like the teacher?
    tb, sb, kl, preds, acts, nb, nw, nn_ = [], [], [], [], [], [], [], []
    for _ in range(n_batches):
        x, _ = make_batch(data, c, bsz, False, dev)
        o = compute(model, x, None, lc)
        tb.append(o["ce_t_future"].item()); sb.append(o["ce_s"].item()); kl.append(o["kl"].item())
        nb.append(compute(model, x, None, lc, memory=False)["ce_s"].item())
        w, n = no_window_ce(model, x)
        nw.append(w); nn_.append(n)
        preds.append(o["div_pred"].flatten()); acts.append(torch.log1p(o["kl_pos"].clamp_min(0)).flatten())
    p, a = torch.cat(preds), torch.cat(acts)
    res["teacher_bpb"] = np.mean(tb) / ln2
    res["student_bpb"] = np.mean(sb) / ln2
    res["no_memory_bpb"] = np.mean(nb) / ln2      # control: recent window only, no latents
    res["no_window_bpb"] = np.mean(nw) / ln2      # latents only: the clearest test that memory carries content
    res["no_window_no_memory_bpb"] = np.mean(nn_) / ln2
    res["kl_nats"] = float(np.mean(kl))
    corr = float(torch.corrcoef(torch.stack([p, a]))[0, 1])
    res["div_corr"] = corr if math.isfinite(corr) else 0.0  # nan if either side is constant
    res["div_pred_natural"] = float(p.mean())

    # copy probes: is exact detail preserved, and does recall find it?
    span, cue = c.future - 1, lc.probe_cue
    acc = {"teacher": [], "no_memory": [], "none": [], "oracle": [], "pointer": []}
    hits, hits_trained, hits_any, div_probe = [], [], [], []

    def copy_acc(logits, yf):
        sel = slice(cue + 1, span + 1)          # logits index m+1 predicts span byte m
        return (logits[:, sel].argmax(-1) == yf[:, sel]).float().mean().item()

    for _ in range(n_probes):
        x, src = make_batch(data, c, 1, True, dev)
        o = compute(model, x, src, lc)
        acc["teacher"].append(copy_acc(o["logit_tf"], o["y_f"]))
        acc["none"].append(copy_acc(o["logit_s"], o["y_f"]))
        acc["no_memory"].append(copy_acc(compute(model, x, src, lc, memory=False)["logit_s"], o["y_f"]))
        div_probe.append(o["div_pred"][:, cue + 1:].mean().item())
        oracle = int((src[0] + cue + (span - cue) // 2) // c.block)
        choice = int(o["ptr_logits"][0, 0].argmax())  # decided after PROBE + cue bytes
        hits.append(float(choice == oracle))
        # the pointer at this query is TRAINED to name the block of the next span byte, (src+cue)//block;
        # a 255-byte span nearly always straddles two blocks, so also count any block the scored span touches
        first, last_b = int((src[0] + cue) // c.block), int((src[0] + span - 1) // c.block)
        hits_trained.append(float(choice == first))
        hits_any.append(float(first <= choice <= last_b))
        acc["oracle"].append(copy_acc(compute(model, x, src, lc, expand_idx=oracle)["logit_s"], o["y_f"]))
        acc["pointer"].append(copy_acc(compute(model, x, src, lc, expand_idx=choice)["logit_s"], o["y_f"]))
    for k, v in acc.items():
        res[f"copy_acc_{k}"] = float(np.mean(v))
    res["pointer_hit"] = float(np.mean(hits))
    res["pointer_hit_trained"] = float(np.mean(hits_trained))
    res["pointer_hit_any"] = float(np.mean(hits_any))
    res["div_pred_probe"] = float(np.mean(div_probe))
    model.train()
    return res


@torch.no_grad()
def attn_mass(model, x_in, pos, q_idx, k_mask):
    """Per layer: mean attention probability (over heads and queries q_idx) landing on keys where k_mask."""
    out, x = [], x_in
    for layer in model.layers:
        B, T, D = x.shape
        q, k, v = layer.qkv(layer.n1(x)).view(B, T, 3, layer.h, D // layer.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos, layer.base), rope(k, pos, layer.base)
        a = (q[:, :, q_idx] @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        a = a.masked_fill(q_idx[:, None] < torch.arange(T, device=x.device)[None], float("-inf")).softmax(-1)
        out.append(float(a[..., k_mask].sum(-1).mean()))
        x = layer(x, pos)
    return out


@torch.no_grad()
def diagnose(model, data, lc: LossCfg, n_batches=8, bsz=8):
    """Is compressed memory used? Norms, attention to memory, and prediction with the recent window removed."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    L_old, W = c.n_blocks * c.block, c.window
    tok_rms = float(model.emb.weight[:256].pow(2).mean(-1).sqrt().mean())
    lat_rms, m_t, m_s, nowin, nothing, full = [], [], [], [], [], []
    for _ in range(n_batches):
        x, _ = make_batch(data, c, bsz, False, dev)
        L = x.shape[1]
        lats = []
        for i in range(c.n_blocks):
            bs = i * c.block
            lats.append(model.compress([(l, j * c.block) for j, l in enumerate(lats)],
                                       x[:, bs:bs + c.block], bs, x[:, bs + c.block:bs + c.block + W], bs + c.block))
        lat_rms.append(float(torch.cat(lats, 1).float().pow(2).mean(-1).sqrt().mean()))
        fut = torch.arange(L_old + W, L - 1, device=dev)               # teacher queries predicting the future
        m_t.append(attn_mass(model, model.emb(x), torch.arange(L, device=dev), fut,
                             torch.arange(L, device=dev) < L_old))
        parts = [("lat", lats[j], j * c.block) for j in range(c.n_blocks)]
        n_lat = c.n_blocks * c.k_lat
        xs = torch.cat([torch.cat(lats, 1).to(model.emb.weight.dtype), model.emb(x[:, L_old:])], 1)
        ps = torch.cat([torch.cat([model.lat_pos(j * c.block, dev) for j in range(c.n_blocks)]),
                        torch.arange(L_old, L, device=dev)])
        T = xs.shape[1]
        m_s.append(attn_mass(model, xs, ps, torch.arange(n_lat + W, T - 1, device=dev),
                             torch.arange(T, device=dev) < n_lat))
        # future with NO recent window: [latents | future] vs [future] alone vs [latents | window | future]
        y = x[:, L_old + W + 1:]
        for store, prts in ((nowin, parts), (nothing, [])):
            h, _, off = model.student(prts, x[:, L_old + W:], L_old + W)
            store.append(F.cross_entropy(model.head(h[:, off:-1]).float().reshape(-1, VOCAB), y.reshape(-1)).item())
        h, _, off = model.student(parts, x[:, L_old:], L_old)
        full.append(F.cross_entropy(model.head(h[:, off + W:-1]).float().reshape(-1, VOCAB), y.reshape(-1)).item())
    ln2 = math.log(2)
    print(f"token embedding RMS     {tok_rms:.4f}")
    print(f"latent RMS              {np.mean(lat_rms):.4f}   (ratio {np.mean(lat_rms) / tok_rms:.1f}x)")
    print("attention from future queries, per layer:")
    print("  teacher -> raw old text  " + " ".join(f"{v:.3f}" for v in np.mean(m_t, 0)))
    print("  student -> latents       " + " ".join(f"{v:.3f}" for v in np.mean(m_s, 0)))
    print("future bpb (bytes 1.., same targets):")
    print(f"  latents + window        {np.mean(full) / ln2:.4f}")
    print(f"  latents, no window      {np.mean(nowin) / ln2:.4f}")
    print(f"  nothing (future only)   {np.mean(nothing) / ln2:.4f}")
    model.train()


# ============================================================================ streaming inference

@torch.no_grad()
def generate(model, prompt: bytes, n_new, temperature=0.8, top_k=50, tau=None, cooldown=32,
             max_blocks=None, max_expanded=1, recompress_every=0, verbose=True, use_cache=True):
    """Streams bytes with three tiers. tau: recall threshold on predicted log1p(KL); None disables recall.
    use_cache: KV-cache the student context. The context layout only changes on eviction, recall and
    re-compression; those rebuild the cache (one prefill), every other byte is a single-token decode."""
    model.eval()
    c, dev = model.c, next(model.parameters()).device
    B, W = c.block, c.window
    max_blocks = max_blocks or c.n_blocks
    blocks = []                      # dict(start, raw (1,B), lat (1,K,D), expanded)
    recent, rstart = list(prompt), 0
    expanded_order, since_recall, since_recompress = [], cooldown, 0
    out, events = [], []

    def ctx():
        return blocks[-max_blocks:]

    def compress_block(raw_t, start, cond, cond_start, prev_blocks, base):
        prev = [(b["lat"], b["start"] - base) for b in prev_blocks[-(c.n_blocks - 1):]] if c.n_blocks > 1 else []
        return model.compress(prev, raw_t, start - base, cond, cond_start - base)

    st = dict(valid=False)           # KV cache state: caches, block summaries, tail bytes cached, positions

    def evict():
        nonlocal recent, rstart
        while len(recent) >= W + B:
            st["valid"] = False
            base = ctx()[0]["start"] if ctx() else rstart
            raw_t = torch.tensor(recent[:B], device=dev)[None]
            cond = torch.tensor(recent[B:B + W], device=dev)[None]
            lat = compress_block(raw_t, rstart, cond, rstart + B, ctx(), base)
            blocks.append(dict(start=rstart, raw=raw_t, lat=lat, expanded=False))
            recent, rstart = recent[B:], rstart + B
            for b in blocks[:-max_blocks]:  # blocks leaving the context are no longer expanded
                b["expanded"] = False

    def recompress():
        # re-encode visible blocks conditioned on the CURRENT recent window (task may have shifted)
        cb = ctx()
        if not cb or len(recent) < 1:
            return
        base = cb[0]["start"]
        cond = torch.tensor(recent[-W:], device=dev)[None]
        cond_start = rstart + len(recent) - cond.shape[1]
        done = []
        for b in cb:
            b["lat"] = compress_block(b["raw"], b["start"], cond, cond_start, done, base)
            done.append(b)
        st["valid"] = False

    def forward():
        """-> hidden state of the newest byte (1, D), block summaries for the pointer (1, n_parts, D) or None."""
        if not (use_cache and st["valid"]):
            cb = ctx()
            base = cb[0]["start"] if cb else rstart
            parts = [("raw", b["raw"], b["start"] - base) if b["expanded"] else ("lat", b["lat"], b["start"] - base)
                     for b in cb]
            tail = torch.tensor(recent, device=dev)[None]
            caches = [{} for _ in model.layers] if use_cache else None
            h, seg, off = model.student(parts, tail, rstart - base, caches)
            summ = model.block_summaries(h[:, :off], seg[:off], len(parts)) if parts else None
            st.update(valid=use_cache, caches=caches, summ=summ, n_cached=len(recent),
                      tail_pos=rstart - base)
            return h[:, -1], summ
        while st["n_cached"] < len(recent):              # append the bytes generated since the last call
            t = torch.tensor([[recent[st["n_cached"]]]], device=dev)
            pos = torch.tensor([st["tail_pos"] + st["n_cached"]], device=dev)
            h = model.run(model.emb(t), pos, st["caches"])
            st["n_cached"] += 1
        return h[:, -1], st["summ"]

    if not recent:
        recent = [10]
    evict()
    while len(out) < n_new:
        last, summ = forward()
        if tau is not None and summ is not None and since_recall >= cooldown:
            div = model.div_head(last).item()
            if div > tau:
                pl = model.pointer_scores(summ, last[:, None])[0, 0]
                cb = ctx()
                for j, b in enumerate(cb):
                    if b["expanded"]:
                        pl[j] = -float("inf")
                if torch.isfinite(pl).any():
                    j = int(pl.argmax())
                    cb[j]["expanded"] = True
                    expanded_order.append(cb[j])
                    while sum(b["expanded"] for b in cb) > max_expanded:
                        old = expanded_order.pop(0)
                        old["expanded"] = False
                    events.append(dict(at=len(out), block_start=cb[j]["start"], div=div))
                    since_recall = 0
                    st["valid"] = False
                    last, summ = forward()
        logits = model.head(last)[0, :256].float() / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, 256))
            logits[logits < v[-1]] = -float("inf")
        nb = int(torch.multinomial(F.softmax(logits, -1), 1))
        out.append(nb)
        recent.append(nb)
        since_recall += 1
        since_recompress += 1
        evict()
        if recompress_every and since_recompress >= recompress_every:
            recompress()
            since_recompress = 0
    if verbose:
        print(f"[blocks stored: {len(blocks)}, in context: {len(ctx())}, recalls: {len(events)}]")
        for e in events[:20]:
            print(f"  recall at byte {e['at']}: block@{e['block_start']} (pred log1p KL {e['div']:.3f})")
    return bytes(out), events


# ============================================================================ training

def train(args):
    dev = pick_device()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    c = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    lc = LossCfg(**{f.name: getattr(args, f.name) for f in fields(LossCfg)})

    data = torch.from_numpy(np.fromfile(args.data, dtype=np.uint8).copy())
    L = c.n_blocks * c.block + c.window + c.future
    n = int(0.95 * len(data))
    train_d, val_d = data[:n], data[n:]
    assert len(train_d) > 4 * L and len(val_d) > 2 * L, "data file too small for this context size"

    model = RollingLatentLM(c).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device {dev} | "
          f"teacher ctx {L} bytes | student ctx {c.n_blocks * c.k_lat + c.window + c.future} "
          f"| compression {c.block // c.k_lat}x per block")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=(dev != "cpu"))  # foreach Adam faults the XPU
    amp = torch.autocast(dev, dtype=torch.bfloat16) if dev != "cpu" else torch.autocast("cpu", enabled=False)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    for step in range(args.steps + 1):
        if (step % args.eval_every == 0 and step > 0) or step == args.steps:
            with amp:
                r = evaluate(model, val_d, lc, n_batches=args.eval_batches, bsz=args.batch,
                             n_probes=args.eval_probes) if step >= lc.distill_start else {}
            print(f"== eval step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
            torch.save({"model": model.state_dict(), "config": asdict(c), "loss_cfg": asdict(lc), "step": step},
                       args.out)
        if step == args.steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        probe = random.random() < args.probe_frac
        expand_idx = random.randrange(c.n_blocks) if random.random() < args.expand_prob else None
        recond = random.random() < args.recond_prob
        drop_window = random.random() < args.drop_window_prob
        x, src = make_batch(train_d, c, args.batch, probe, dev)
        with amp:
            o = compute(model, x, src, lc, expand_idx=expand_idx, recond=recond, step=step, drop_window=drop_window)
        o["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)  # _foreach_* faults the XPU
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            mem = f" | {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev == "cuda" else ""
            dt = (time.time() - t0) / args.log_every * 1000
            t0 = time.time()
            keys = [k for k in ("loss", "lm", "ce_s", "kl", "div", "ptr") if k in o]
            print(f"step {step} | " + " ".join(f"{k} {o[k].item():.4f}" for k in keys) +
                  f" | gn {gn:.2f} | lr {lr_at(step):.2e} | {'probe ' if probe else ''}"
                  f"{'exp' + str(expand_idx) + ' ' if expand_idx is not None else ''}"
                  f"{'recond ' if recond else ''}{'nowin ' if drop_window else ''}| {dt:.0f} ms/step{mem}")


def load(path):
    dev = pick_device()
    ck = torch.load(path, map_location=dev)
    model = RollingLatentLM(Config(**{"lat_norm": False, **ck["config"]})).to(dev)  # older ckpts: no lat_norm
    model.load_state_dict(ck["model"])
    return model.eval(), LossCfg(**ck["loss_cfg"])


# ============================================================================ selftest

def selftest():
    """Tiny model on CPU/GPU. Checks shapes, causality, information flow, gradients, overfitting, streaming."""
    torch.manual_seed(0); random.seed(0)
    dev = pick_device()
    c = Config(d_model=64, n_layers=2, n_heads=4, block=32, k_lat=4, n_blocks=3, window=32, future=32, ptr_dim=16)
    lc = LossCfg(distill_start=0, kl_ramp=1)
    model = RollingLatentLM(c).to(dev)
    text = ("the quick brown fox jumps over the lazy dog. " * 50 + "lorem ipsum dolor sit amet 0123456789. " * 50)
    data = torch.tensor(list(text.encode()), dtype=torch.uint8)
    L_old = c.n_blocks * c.block
    L = L_old + c.window + c.future
    ok = lambda name: print(f"  [ok] {name}")

    # 1. teacher causality
    x = torch.randint(0, 256, (1, L), device=dev)
    with torch.no_grad():
        h1 = model.run(model.emb(x), torch.arange(L, device=dev))
        x2 = x.clone(); x2[0, 100] = (x2[0, 100] + 1) % 256
        h2 = model.run(model.emb(x2), torch.arange(L, device=dev))
    assert torch.allclose(h1[:, :100], h2[:, :100], atol=1e-5) and not torch.allclose(h1[:, 100:], h2[:, 100:])
    ok("teacher is causal")

    # 2. student causality and information flow through latents
    def student_hidden(xx, expand=None):
        lats = []
        for i in range(c.n_blocks):
            bs, cs = i * c.block, (i + 1) * c.block
            lats.append(model.compress([(l, j * c.block) for j, l in enumerate(lats)],
                                       xx[:, bs:bs + c.block], bs, xx[:, cs:cs + c.window], cs))
        parts = [("raw", xx[:, j * c.block:(j + 1) * c.block], j * c.block) if j == expand else
                 ("lat", lats[j], j * c.block) for j in range(c.n_blocks)]
        return model.student(parts, xx[:, L_old:], L_old)

    with torch.no_grad():
        hs1, seg, off = student_hidden(x)
        assert hs1.shape[1] == c.n_blocks * c.k_lat + c.window + c.future
        j = c.window + 5                                  # a byte inside the future region
        x3 = x.clone(); x3[0, L_old + j] = (x3[0, L_old + j] + 1) % 256
        hs3, _, _ = student_hidden(x3)
        assert torch.allclose(hs1[:, :off + j], hs3[:, :off + j], atol=1e-5)
        ok("student is causal w.r.t. the future region")
        x4 = x.clone(); x4[0, 5] = (x4[0, 5] + 1) % 256  # a byte inside compressed block 0
        hs4, _, _ = student_hidden(x4)
        assert not torch.allclose(hs1[:, off + c.window:], hs4[:, off + c.window:], atol=1e-6)
        ok("old-block content reaches student predictions through latents")
        hs5, seg5, off5 = student_hidden(x, expand=1)
        assert hs5.shape[1] == hs1.shape[1] + c.block - c.k_lat and (seg5 == 1).sum() == c.block
        ok("expanded layout has the right shape")

    # 3. all losses finite, gradients reach every new component
    for probe in (False, True):
        for expand in (None, 1):
            for recond in (False, True):
                model.zero_grad()
                xb, src = make_batch(data, c, 4, probe, dev)
                o = compute(model, xb, src, lc, expand_idx=expand, recond=recond, step=10)
                assert torch.isfinite(o["loss"]), (probe, expand, recond)
                o["loss"].backward()
    for probe in (False, True):
        xb, src = make_batch(data, c, 4, probe, dev)
        o = compute(model, xb, src, lc, expand_idx=1, step=10, drop_window=True)
        assert torch.isfinite(o["loss"]), ("drop_window", probe)
        o["loss"].backward()
    for name in ("beacons", "lat_type"):
        assert getattr(model, name).grad is not None and getattr(model, name).grad.abs().sum() > 0, name
    for mod in ("adapter", "div_head", "ptr_q", "ptr_k"):
        g = sum(p.grad.abs().sum().item() for p in getattr(model, mod).parameters() if p.grad is not None)
        assert g > 0, mod
    ok("losses finite in all 8 batch modes + window dropout; gradients reach compressor, div head and pointer")

    # 4. overfit a fixed probe batch
    xb, src = make_batch(data, c, 8, True, dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, fused=(dev != "cpu"))  # foreach Adam faults the XPU
    first = None
    for s in range(200):
        o = compute(model, xb, src, lc, step=10)
        if first is None:
            first = o["loss"].item()
        opt.zero_grad(); o["loss"].backward(); opt.step()
    last = o["loss"].item()
    assert last < 0.7 * first, (first, last)
    ok(f"overfits a fixed batch: loss {first:.3f} -> {last:.3f}")

    # 5. evaluation and streaming generation with forced recall and re-compression
    r = evaluate(model, data, lc, n_batches=2, bsz=2, n_probes=3)
    assert all(np.isfinite(v) for v in r.values()), r
    ok("evaluate runs: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items()))
    out, events = generate(model, text[:300].encode(), 200, tau=-1e9, cooldown=16,
                           recompress_every=50, verbose=False)
    assert len(out) == 200 and len(events) > 0
    ok(f"streaming generation with {len(events)} recalls and periodic re-compression")
    gen = lambda cache: generate(model, text[:300].encode(), 300, top_k=1, tau=0.0, cooldown=16,
                                 recompress_every=70, verbose=False, use_cache=cache)
    (o1, e1), (o2, e2) = gen(False), gen(True)
    assert o1 == o2 and [e["at"] for e in e1] == [e["at"] for e in e2], (o1, o2, e1, e2)
    ok(f"KV-cached generation matches uncached ({len(e1)} recalls, greedy, 300 bytes)")
    print("SELFTEST PASSED")


# ============================================================================ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="rolling_latent_lm.pt")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=6e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--probe_frac", type=float, default=0.5)
    t.add_argument("--expand_prob", type=float, default=0.3)
    t.add_argument("--recond_prob", type=float, default=0.2)
    t.add_argument("--drop_window_prob", type=float, default=0.25)
    t.add_argument("--eval_every", type=int, default=1000)
    t.add_argument("--eval_batches", type=int, default=10)
    t.add_argument("--eval_probes", type=int, default=32)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    for f in fields(Config) + fields(LossCfg):
        if f.type in (bool, "bool"):
            t.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            typ = {"int": int, "float": float}.get(f.type, f.type)
            t.add_argument(f"--{f.name}", type=typ, default=f.default)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--batches", type=int, default=20)
    e.add_argument("--probes", type=int, default=64)

    d = sub.add_parser("diag")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--data", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompt", default="\n")
    g.add_argument("--prompt_file", default=None)
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k", type=int, default=50)
    g.add_argument("--tau", type=float, default=None, help="recall threshold on predicted log1p(KL)")
    g.add_argument("--cooldown", type=int, default=32)
    g.add_argument("--max_expanded", type=int, default=1)
    g.add_argument("--recompress_every", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        model, lc = load(args.ckpt)
        data = torch.from_numpy(np.fromfile(args.data, dtype=np.uint8).copy())
        data = data[int(0.95 * len(data)):]
        r = evaluate(model, data, lc, n_batches=args.batches, n_probes=args.probes)
        for k, v in r.items():
            print(f"{k:20s} {v:.4f}")
    elif args.cmd == "diag":
        model, lc = load(args.ckpt)
        data = torch.from_numpy(np.fromfile(args.data, dtype=np.uint8).copy())
        diagnose(model, data[int(0.95 * len(data)):], lc)
    else:
        torch.manual_seed(args.seed)
        model, _ = load(args.ckpt)
        prompt = open(args.prompt_file, "rb").read() if args.prompt_file else args.prompt.encode()
        out, _ = generate(model, prompt, args.n, args.temperature, args.top_k, args.tau, args.cooldown,
                          max_expanded=args.max_expanded, recompress_every=args.recompress_every)
        print(out.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
