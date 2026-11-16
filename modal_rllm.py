"""
Run rolling_latent_lm.py on a Modal GPU.

One-time setup:
    pip install modal
    modal setup                      # opens a browser to log in

Data (downloads enwik8 into the Modal volume "rllm", once):
    modal run modal_rllm.py::get_data

Run any rolling_latent_lm.py command (--script qwen: rolling_latent_qwen.py); everything after --args is passed through.
Paths are relative to the volume, so checkpoints persist between runs:
    modal run modal_rllm.py --args "selftest"
    modal run modal_rllm.py --gpu L40S --args "train --data enwik8 --out smoke.pt --steps 2000 \
        --distill_start 300 --kl_ramp 300 --eval_every 500 --d_model 256 --n_layers 4 --n_heads 4"
    modal run --detach modal_rllm.py --gpu H100 --args "train --data enwik8 --out run1.pt --steps 30000"

--detach keeps the job running if your laptop sleeps or the terminal closes. Watch it with
`modal app logs rolling-latent-lm` or on the dashboard (modal.com/apps).

Qwen port (model weights are cached in the volume under hf/):
    modal run modal_rllm.py --script qwen --gpu cpu --online --args "prep --data enwik8 --out enwik8.qwen3.npy"
    modal run modal_rllm.py --script qwen --args "train --tokens enwik8.qwen3.npy --out q06.pt --steps 200"

Logs are also written to the volume as logs/<out>.log. Fetch results with
    modal volume get rllm run1.pt .
    modal volume get rllm logs/run1.log .
"""
import shlex
import subprocess
import sys

import modal

app = modal.App("rolling-latent-lm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers==5.23.0", "pyarrow", "flash-linear-attention")
    # causal-conv1d is not installed: it is a CUDA extension and no wheel matches this torch; building it needs a
    # CUDA devel image. It only replaces a 4-tap depthwise conv, so the PyTorch fallback costs little.
    .env({"HF_HOME": "/vol/hf"})       # model downloads are cached in the volume
    .add_local_file("rolling_latent_lm.py", "/root/rolling_latent_lm.py")
    .add_local_file("rolling_latent_qwen.py", "/root/rolling_latent_qwen.py")
    .add_local_file("cram.py", "/root/cram.py")
    .add_local_file("rolling_latent_chat.py", "/root/rolling_latent_chat.py")
)
SCRIPTS = {"byte": "rolling_latent_lm.py", "qwen": "rolling_latent_qwen.py", "cram": "cram.py",
           "chat": "rolling_latent_chat.py"}
vol = modal.Volume.from_name("rllm", create_if_missing=True)


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def get_data():
    import os, urllib.request, zipfile
    if os.path.exists("/vol/enwik8"):
        print("enwik8 already present")
        return
    urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", "/tmp/enwik8.zip")
    zipfile.ZipFile("/tmp/enwik8.zip").extract("enwik8", "/vol")
    vol.commit()
    print("enwik8:", os.path.getsize("/vol/enwik8"), "bytes")


@app.function(image=image, gpu="L40S", timeout=600)
def check_kernels():
    """Which optional fast kernels load in the image (modal run modal_rllm.py::check_kernels)."""
    import importlib
    import torch
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    for m in ("fla", "causal_conv1d"):
        try:
            mod = importlib.import_module(m)
            print(f"{m} OK {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"{m} not available: {type(e).__name__}: {str(e)[:200]}")


@app.function(image=image, volumes={"/vol": vol}, timeout=24 * 3600)
def run(argv: list[str], script: str = "byte", online: bool = False):
    import os
    os.makedirs("/vol/logs", exist_ok=True)
    name = argv[argv.index("--out") + 1] if "--out" in argv else argv[0]
    log_path = f"/vol/logs/{os.path.splitext(os.path.basename(name))[0]}.log"
    cmd = [sys.executable, "-u", f"/root/{SCRIPTS[script]}", *argv]
    print("$", shlex.join(cmd), flush=True)
    import threading

    def autocommit():  # persist checkpoints/logs every 2 min, in case the job dies mid-run
        while True:
            threading.Event().wait(120)
            vol.commit()
    threading.Thread(target=autocommit, daemon=True).start()
    with open(log_path, "a") as log:
        # offline by default: models are cached in /vol/hf, and anonymous Hub API calls get rate-limited (429)
        env = dict(os.environ, **({} if online else {"HF_HUB_OFFLINE": "1"}))
        p = subprocess.Popen(cmd, cwd="/vol", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        p.wait()
    vol.commit()
    if p.returncode:
        raise SystemExit(p.returncode)


@app.local_entrypoint()
def main(args: str = "selftest", gpu: str = "L40S", script: str = "byte", online: bool = False):
    """--gpu cpu runs without a GPU (e.g. tokenisation). --online allows Hugging Face downloads (first use of
    a model); otherwise the Hub is offline and only the cache in the volume is used."""
    opts = dict(cpu=4, memory=16384) if gpu == "cpu" else dict(gpu=gpu)
    run.with_options(**opts).remote(shlex.split(args), script, online)
