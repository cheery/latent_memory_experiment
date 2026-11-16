#!/usr/bin/env python3
"""
cram.py: how much text can a frozen LLM reconstruct from a few input vectors?

Replicates the per-sample optimisation of Kuratov et al. (2025, "Cramming 1568 tokens into a single vector
and back again") on our base model. For each text span x_1..x_N, m "memory" vectors are optimised by
gradient descent (the model stays frozen) so that the model reading [mem_1..mem_m] reproduces the span:

    input  [mem_1 .. mem_m, x_1 .. x_{N-1}]      targets  x_1 .. x_N   (teacher forcing)

This is the CEILING for compression into latents: an encoder that produces latents in one pass can at best
find what this optimisation finds. If teacher-forced argmax accuracy is 100%, greedy decoding from the
memory vectors alone reproduces the span exactly, i.e. "recall" could regenerate the text from latents.

Reported per (m, N):
    acc          teacher-forced argmax token accuracy (mean over samples)
    perfect      fraction of samples reconstructed exactly
    prefix       tokens reproduced before the first error = what greedy decoding would recover
    gain_bits    information gain: CE(text | nothing) - CE(text | mem), summed over x_2..x_N, in bits
    steps        optimisation steps used (stops early once every sample is perfect)

Commands:
    python cram.py selftest                                  # tiny random Qwen3, no download
    python cram.py run --tokens enwik8.qwen3.npy --lengths 128 512 1024 --mems 1 4
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def pick_device():
    d = os.environ.get("RLLM_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    return "xpu" if torch.xpu.is_available() else "cpu"


def hidden(lm, x, pos):
    B, T = x.shape[:2]
    return lm.model(inputs_embeds=x, position_ids=pos[None].expand(B, -1),
                    attention_mask=torch.ones(B, T, dtype=torch.long, device=x.device), use_cache=False).last_hidden_state


def chunked_ce(lm, h, y, chunk=256):
    """Per-token CE (B, N) without materialising (B, N, vocab) logits for backward: one chunk at a time."""
    def f(hh, yy):
        lg = lm.lm_head(hh).float()
        return F.cross_entropy(lg.flatten(0, 1), yy.flatten(), reduction="none").view(yy.shape), \
            (lg.argmax(-1) == yy).float()
    ces, hits = [], []
    for i in range(0, y.shape[1], chunk):
        ce, hit = checkpoint(f, h[:, i:i + chunk], y[:, i:i + chunk], use_reentrant=False) if h.requires_grad \
            else f(h[:, i:i + chunk], y[:, i:i + chunk])
        ces.append(ce); hits.append(hit)
    return torch.cat(ces, 1), torch.cat(hits, 1)


def no_mem_bits(lm, x):
    """Per-sample bits for x_2..x_N with no memory (the information-gain baseline)."""
    with torch.no_grad():
        N = x.shape[1]
        h0 = hidden(lm, lm.get_input_embeddings()(x[:, :-1]), torch.arange(N - 1, device=x.device))
        ce0, _ = chunked_ce(lm, h0, x[:, 1:])
    return ce0.sum(1) / math.log(2)


def decode_ce(lm, mem, x):
    """Frozen model reading [mem | x_1..x_{N-1}] -> per-token CE and argmax hits for x_1..x_N."""
    m, N = mem.shape[1], x.shape[1]
    emb = lm.get_input_embeddings()
    inp = torch.cat([mem.to(emb.weight.dtype), emb(x[:, :-1])], 1)
    h = hidden(lm, inp, torch.arange(m + N - 1, device=x.device))[:, m - 1:]
    return chunked_ce(lm, h, x)


def summarize(lm, x, hit, ce, base_bits):
    N = x.shape[1]
    acc = hit.mean(1)
    first_err = torch.where(hit < 1, torch.arange(N, device=x.device)[None].expand_as(hit), N).min(1).values
    return dict(acc=acc.mean().item(), perfect=(acc == 1).float().mean().item(), prefix=first_err.float().mean().item(),
                gain_bits=(base_bits - ce[:, 1:].sum(1) / math.log(2)).mean().item(),
                bits_no_mem=base_bits.mean().item())


@torch.no_grad()
def measure(lm, mem, x):
    """Metrics for given memory vectors, no optimisation."""
    ce, hit = decode_ce(lm, mem, x)
    return summarize(lm, x, hit, ce, no_mem_bits(lm, x))


def cram(lm, x, m, steps=3000, lr=1e-2, log=None, init=None):
    """x: (S, N) token ids. Optimises m vectors per sample (from init (S,m,D) if given, else random at
    token-embedding scale). Returns a dict of metrics (best over steps)."""
    S, N = x.shape
    dev = x.device
    emb = lm.get_input_embeddings()
    dt = emb.weight.dtype
    rms = emb.weight.float().pow(2).mean().sqrt()
    mem = torch.nn.Parameter(init.detach().float().clone() if init is not None
                             else torch.randn(S, m, emb.weight.shape[1], device=dev) * rms)
    if steps == 0:
        return dict(measure(lm, mem, x), steps=0, mem_rms_ratio=(mem.pow(2).mean().sqrt() / rms).item())
    opt = torch.optim.Adam([mem], lr=lr)
    with torch.no_grad():
        tok = emb(x[:, :-1])
    pos = torch.arange(m + N - 1, device=dev)
    best_acc = torch.zeros(S, device=dev)
    best_prefix = torch.zeros(S, device=dev)
    best_ce = torch.full((S,), float("inf"), device=dev)
    step = 0
    for step in range(1, steps + 1):
        h = hidden(lm, torch.cat([mem.to(dt), tok], 1), pos)[:, m - 1:]      # positions predicting x_1..x_N
        ce, hit = chunked_ce(lm, h, x)
        loss = ce.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = hit.mean(1)
            first_err = torch.where(hit < 1, torch.arange(N, device=dev)[None].expand(S, -1), N).min(1).values
            better = acc > best_acc
            best_acc = torch.where(better, acc, best_acc)
            best_prefix = torch.maximum(best_prefix, first_err.float())
            best_ce = torch.minimum(best_ce, ce[:, 1:].sum(1))
        if log and (step % log == 0 or step == 1):
            print(f"    step {step}: loss {loss.item():.4f} acc {acc.mean().item():.4f} "
                  f"perfect {(acc == 1).float().mean().item():.2f}", flush=True)
        if bool((best_acc == 1).all()):
            break
    with torch.no_grad():                                                # baseline: the text with no memory
        h0 = hidden(lm, tok, torch.arange(N - 1, device=dev))
        ce0, _ = chunked_ce(lm, h0, x[:, 1:])
    gain = (ce0.sum(1) - best_ce) / math.log(2)
    return dict(acc=best_acc.mean().item(), perfect=(best_acc == 1).float().mean().item(),
                prefix=best_prefix.mean().item(), gain_bits=gain.mean().item(),
                bits_no_mem=(ce0.sum(1) / math.log(2)).mean().item(), steps=step,
                mem_rms_ratio=(mem.detach().pow(2).mean().sqrt() / rms).item())


def run(args):
    from transformers import AutoModelForCausalLM
    dev = pick_device()
    torch.manual_seed(args.seed)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    tokens = torch.from_numpy(np.load(args.tokens).astype(np.int64))
    tokens = tokens[int(0.95 * len(tokens)):]                               # held-out text, as in training
    print(f"model {args.model} | samples {args.samples} | max steps {args.steps} | lr {args.lr}", flush=True)
    print(f"{'m':>3} {'N':>5} {'acc':>7} {'perfect':>8} {'prefix':>8} {'gain_bits':>10} {'bits_no_mem':>12} "
          f"{'steps':>6} {'tok/vec':>8} {'time':>6}", flush=True)
    rows = []
    for m in args.mems:
        for N in args.lengths:
            g = torch.Generator().manual_seed(args.seed + N)
            ix = torch.randint(0, len(tokens) - N, (args.samples,), generator=g)
            x = torch.stack([tokens[i:i + N] for i in ix.tolist()]).to(dev)
            t0 = time.time()
            r = cram(lm, x, m, args.steps, args.lr, log=args.log_every or None)
            print(f"{m:>3} {N:>5} {r['acc']:>7.4f} {r['perfect']:>8.2f} {r['prefix']:>8.1f} {r['gain_bits']:>10.1f} "
                  f"{r['bits_no_mem']:>12.1f} {r['steps']:>6} {r['prefix'] / m:>8.1f} {time.time() - t0:>5.0f}s",
                  flush=True)
            rows.append(dict(m=m, N=N, seconds=time.time() - t0, **r))
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(dict(model=args.model, samples=args.samples, max_steps=args.steps, lr=args.lr,
                                   rows=rows), f, indent=1)


# ============================================================================ Phase B: a trained encoder

class Encoder(torch.nn.Module):
    """One forward pass of the base model + LoRA over [x_1..x_N | k beacons] -> k latents for the FROZEN base
    model (LoRA off) to decode. Latents are RMS-normalised to token-embedding scale times a learned scale
    (Phase A's optimised vectors are unconstrained, so the scale is free to grow)."""

    def __init__(self, lm, k, r=16, alpha=32.0, init_scale=1.0):
        super().__init__()
        from rolling_latent_qwen import LoRALinear, RMSNorm
        self.lm, self.k = lm, k
        self.loras = []
        for layer in lm.model.layers:
            for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                  (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
                for n in names:
                    lo = LoRALinear(getattr(parent, n), r, alpha)
                    setattr(parent, n, lo)
                    self.loras.append(lo)
        emb = lm.get_input_embeddings().weight
        dev, D = emb.device, emb.shape[1]
        self.register_buffer("emb_rms", emb.float().pow(2).mean().sqrt().detach())
        self.beacons = torch.nn.Parameter(torch.randn(k, D, device=dev) * float(self.emb_rms))
        self.adapter = torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.GELU(), torch.nn.Linear(D, D)).to(dev)
        for p in self.adapter.parameters():
            torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.zeros_(p)
        self.norm = RMSNorm(D).to(dev)
        # Phase A's optimised vectors end up ~3.3x token-embedding RMS (16 x 256 run); at lr 2e-4 a scale starting
        # at 1 barely moves (1.00 -> 1.05 in 1000 steps), so start it where the optimum is and give it its own lr
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(init_scale), device=dev))
        self.set_lora(False)                    # the base model is the decoder unless encoding

    def set_lora(self, on):
        for lo in self.loras:
            lo.enabled = on

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    def forward(self, x):
        S, N = x.shape
        emb = self.lm.get_input_embeddings()
        inp = torch.cat([emb(x), self.beacons.to(emb.weight.dtype)[None].expand(S, -1, -1)], 1)
        self.set_lora(True)
        try:
            h = hidden(self.lm, inp, torch.arange(N + self.k, device=x.device))[:, -self.k:].float()
        finally:
            self.set_lora(False)
        return self.norm(self.adapter(h)) * self.emb_rms * self.log_scale.exp()


def enc_eval(lm, enc, x, refine=(0,), lr=1e-2):
    """Encoder latents as they are (refine 0) and after R steps of Phase-A optimisation started from them."""
    with torch.no_grad():
        lat = enc(x)
    return {R: cram(lm, x, enc.k, steps=R, lr=lr, init=lat) for R in refine}


def enc_train(args):
    from transformers import AutoModelForCausalLM
    dev = pick_device()
    torch.manual_seed(args.seed)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    enc = Encoder(lm, args.k, args.lora_r, init_scale=args.init_scale)
    tokens = torch.from_numpy(np.load(args.tokens).astype(np.int64))
    n = int(0.95 * len(tokens))
    train_t, val_t = tokens[:n], tokens[n:]
    g = torch.Generator().manual_seed(args.seed + args.N)                  # same held-out spans as `run`
    ix = torch.randint(0, len(val_t) - args.N, (args.eval_samples,), generator=g)
    x_val = torch.stack([val_t[i:i + args.N] for i in ix.tolist()]).to(dev)
    params = [p for p in enc.parameters() if p.requires_grad]
    print(f"model {args.model} | k {args.k} | N {args.N} ({args.N / args.k:.0f} tok/latent) | trainable "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    lora_p = [p for lo in enc.loras for p in (lo.A, lo.B)]
    head_p = [p for p in params if all(p is not q for q in lora_p) and p is not enc.log_scale]
    groups = [dict(params=lora_p, lr=args.lr, base=args.lr),
              dict(params=head_p, lr=args.lr_head or args.lr, base=args.lr_head or args.lr),
              dict(params=[enc.log_scale], lr=args.lr_scale, base=args.lr_scale)]
    opt = torch.optim.AdamW(groups, weight_decay=0.0, fused=(dev != "cpu"))
    rows = []

    def report(step, refine):
        res = enc_eval(lm, enc, x_val, refine)
        for R, r in res.items():
            print(f"== eval step {step} refine {R}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
            rows.append(dict(step=step, refine=R, **r))
        if args.out:
            torch.save({"trainable": enc.trainable_state(), "args": vars(args), "step": step}, args.out)
            with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
                json.dump(rows, f, indent=1)

    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 and step:
            report(step, args.refine if step == args.steps else (0,))
            t0 = time.time()
        if step == args.steps:
            break
        f = min(1.0, (step + 1) / args.warmup)
        if args.cosine:
            f *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, args.steps))))
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * f
        ix = torch.randint(0, len(train_t) - args.N, (args.batch,))
        x = torch.stack([train_t[i:i + args.N] for i in ix.tolist()]).to(dev)
        ce, hit = decode_ce(lm, enc(x), x)
        loss = ce.mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            dt = (time.time() - t0) / (args.log_every if step else 1) * 1000
            t0 = time.time()
            print(f"step {step} | loss {loss.item():.4f} acc {hit.mean().item():.4f} | gn {gn:.2f} | "
                  f"scale {enc.log_scale.exp().item():.2f} | {dt:.0f} ms/step", flush=True)


def selftest():
    """Tiny random Qwen3: memory vectors must make a random span far more predictable than without them."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    dev = pick_device()
    # hidden 64 / 2 layers is too small to hold 16 random tokens in 2 vectors (acc ~0.16); 256 / 4 gets them all
    qc = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096)
    lm = Qwen3ForCausalLM(qc).float().to(dev)
    lm.config._attn_implementation = "sdpa"
    for p in lm.parameters():
        p.requires_grad_(False)
    x = torch.randint(0, 512, (2, 16), device=dev)
    r = cram(lm, x, m=2, steps=400, lr=5e-2)
    print(f"  {r}")
    assert r["perfect"] == 1.0 and r["prefix"] == 16 and r["gain_bits"] > 20, r
    assert all(not p.requires_grad for p in lm.parameters())
    print("  [ok] memory vectors carry the span into a frozen model (chunked CE, early stop, baseline)")

    ref = lm.lm_head(hidden(lm, lm.get_input_embeddings()(x), torch.arange(16, device=dev))).detach()
    enc = Encoder(lm, k=2, r=4)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=3e-3)
    xs = torch.randint(0, 512, (4, 16), device=dev)                    # a fixed small set to fit
    first = None
    for _ in range(300):
        ce, _ = decode_ce(lm, enc(xs), xs)
        first = first if first is not None else ce.mean().item()
        opt.zero_grad(); ce.mean().backward(); opt.step()
    r0 = enc_eval(lm, enc, xs, refine=(0, 100))
    print(f"  encoder loss {first:.3f} -> {ce.mean().item():.3f}; refine 0: acc {r0[0]['acc']:.3f}, "
          f"refine 100: acc {r0[100]['acc']:.3f}")
    # random tokens through a tiny random LoRA'd model are hard to encode in one pass: expect only a modest drop
    # (observed 6.24 -> 5.48); refinement from the encoder's latents should then close most of the gap
    assert ce.mean().item() < 0.92 * first and r0[100]["acc"] > r0[0]["acc"]
    after = lm.lm_head(hidden(lm, lm.get_input_embeddings()(x), torch.arange(16, device=dev)))
    assert torch.allclose(ref, after, atol=1e-5), "decoder must stay the base model (LoRA off)"
    print("  [ok] encoder learns to write decodable latents; refinement starts from them; decoder is the base model")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--tokens", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    r.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    r.add_argument("--mems", type=int, nargs="+", default=[1, 4])
    r.add_argument("--samples", type=int, default=8)
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--lr", type=float, default=1e-2)
    r.add_argument("--log_every", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="write the results table as JSON")
    e = sub.add_parser("enc_train", help="Phase B: train an encoder whose latents the frozen model decodes")
    e.add_argument("--tokens", required=True)
    e.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    e.add_argument("--N", type=int, default=256)
    e.add_argument("--k", type=int, default=16)
    e.add_argument("--lora_r", type=int, default=16)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--steps", type=int, default=2000)
    e.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    e.add_argument("--lr_head", type=float, default=0.0, help="adapter + beacons lr (0: same as --lr)")
    e.add_argument("--lr_scale", type=float, default=2e-4, help="lr of the latent log-scale")
    e.add_argument("--init_scale", type=float, default=1.0, help="initial latent RMS / token-embedding RMS")
    e.add_argument("--cosine", action="store_true", help="cosine decay to 10%% over the run")
    e.add_argument("--warmup", type=int, default=100)
    e.add_argument("--eval_every", type=int, default=250)
    e.add_argument("--eval_samples", type=int, default=8)
    e.add_argument("--refine", type=int, nargs="+", default=[0, 50, 200, 1000])
    e.add_argument("--log_every", type=int, default=25)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    {"selftest": lambda a: selftest(), "run": run, "enc_train": enc_train}[args.cmd](args)


if __name__ == "__main__":
    main()
