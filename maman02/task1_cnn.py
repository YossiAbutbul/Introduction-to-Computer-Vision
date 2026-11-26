"""
task1_cnn.py - Task 1.1: my CNN trained from scratch, cat vs dog.
Trains PetCNN, picks the epoch with the lowest val loss and evaluates it
on the test set. Results in OUT_ROOT/task1_cnn.
"""

import json

import numpy as np
import torch
import torch.nn as nn

from utils import config
from utils.data import make_loaders
from utils.metrics import binary_metrics
from utils.models import PetCNN
from utils.plots import confusion, learning_curves, roc_pr
from utils.train import count_flops, count_params, fit, predict, set_seed

EPOCHS = 50
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 0.05


def main():
    out = config.OUT_ROOT / "task1_cnn"
    set_seed(0)
    loaders = make_loaders("species", batch_size=BATCH, augment=True)

    model = PetCNN(n_out=1)
    n_params, flops = count_params(model), count_flops(model)
    print(f"parameters {n_params:,}   FLOPs per image {flops / 1e9:.3f} G")

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = fit(model, loaders, loss_fn, opt, EPOCHS, out, sched)

    # test only once, with the best weights
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    p_dog, y = predict(model, loaders["test"])
    np.savez(out / "test_scores.npz", p_dog=p_dog, y=y)
    m = binary_metrics(p_dog, y)
    m.update(params=n_params, flops=flops, best_epoch=hist["best_epoch"],
             epochs=EPOCHS, train_time_s=hist["train_time_s"],
             peak_gpu_mem_mb=hist.get("peak_gpu_mem_mb"), gpu=hist.get("gpu"))
    (out / "metrics.json").write_text(json.dumps(m, indent=1))

    learning_curves(hist, out / "curves.png", "my CNN - cat vs dog")
    confusion(m["confusion_matrix"], ["cat", "dog"], out / "confusion.png", "my CNN - test")
    roc_pr({"my CNN": (p_dog, y)}, out / "roc.png", out / "pr.png")

    print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=1))
    print("confusion (rows true cat/dog):", m["confusion_matrix"])
    print("saved to", out)


if __name__ == "__main__":
    main()
