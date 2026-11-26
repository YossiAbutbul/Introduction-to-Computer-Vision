"""
train.py - training loop, prediction, parameter and FLOP count.
Works for one logit (cat/dog) and for several classes (breeds).
"""

import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_binary(model_out):
    return model_out.dim() == 1


def run_epoch(model, loader, loss_fn, optimizer=None, scaler=None):
    """One pass over the loader. Trains if an optimizer is given.
    Returns mean loss and accuracy."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        # float16 on the GPU: faster and less memory
        with torch.set_grad_enabled(train), \
                torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out = model(x)
            if is_binary(out):
                loss = loss_fn(out, y.float())
                pred = (out > 0).long()
            else:
                loss = loss_fn(out, y)
                pred = out.argmax(1)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        correct += (pred == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def fit(model, loaders, loss_fn, optimizer, epochs, out_dir, scheduler=None):
    """Train for some epochs and keep the weights of the epoch with the lowest
    validation loss (best.pt). The test set is not used here.
    Saves history.json with loss/accuracy per epoch, time and peak GPU memory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(DEVICE)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time")}
    best_loss = float("inf")
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, loaders["train"], loss_fn, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"], loss_fn)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, lr, dt)):
            hist[k].append(v)
        mark = ""
        if va_loss < best_loss:
            best_loss = va_loss
            hist["best_epoch"] = ep
            torch.save(model.state_dict(), out_dir / "best.pt")
            mark = " *"
        print(f"epoch {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}  lr {lr:.2e}  {dt:.1f}s{mark}",
              flush=True)

    hist["train_time_s"] = time.time() - t_start
    if DEVICE.type == "cuda":
        hist["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        hist["gpu"] = torch.cuda.get_device_name(0)
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    return hist


@torch.no_grad()
def predict(model, loader):
    """Scores for every image: P(dog) for one logit, softmax for classes."""
    model.to(DEVICE).eval()
    probs, labels = [], []
    for x, y in loader:
        out = model(x.to(DEVICE)).float()
        p = torch.sigmoid(out) if is_binary(out) else torch.softmax(out, 1)
        probs.append(p.cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, img_size=224):
    """FLOPs of one forward pass on one image (a multiply-add = 2 FLOPs)."""
    model = copy.deepcopy(model).to("cpu").eval()
    x = torch.zeros(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops()


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
