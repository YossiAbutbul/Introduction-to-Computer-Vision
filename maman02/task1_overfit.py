"""
task1_overfit.py - Task 1.2: overfitting on purpose with the same CNN.
Changes from task 1.1: 10 train images per breed (370), no augmentation,
no dropout, no weight decay, constant learning rate, more epochs.
The validation set is the full one from the split. Results in OUT_ROOT/task1_overfit.
"""

import json

import torch
import torch.nn as nn

from utils import config
from utils.data import get_split, make_loaders, subset_per_breed
from utils.models import PetCNN
from utils.plots import learning_curves
from utils.train import fit, set_seed

N_PER_BREED = 10
EPOCHS = 60
BATCH = 32
LR = 1e-3


def main():
    out = config.OUT_ROOT / "task1_overfit"
    set_seed(0)
    train, _ = get_split()
    small = subset_per_breed(train, N_PER_BREED)
    loaders = make_loaders("species", batch_size=BATCH, augment=False, train_records=small)
    n_dogs = sum(r[2] for r in small)
    print(f"train images {len(small)} (dogs {n_dogs}), val images {len(loaders['val'].dataset)}")

    model = PetCNN(n_out=1, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    hist = fit(model, loaders, nn.BCEWithLogitsLoss(), opt, EPOCHS, out)

    # overfitting starts where the val loss is lowest
    start = hist["best_epoch"]
    i, last = start - 1, EPOCHS - 1
    summary = {
        "train_images": len(small), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "overfit_start_epoch": start,
        "at_start": {k: hist[k][i] for k in ("train_loss", "train_acc", "val_loss", "val_acc")},
        "at_end": {k: hist[k][last] for k in ("train_loss", "train_acc", "val_loss", "val_acc")},
        "best_val_acc": max(hist["val_acc"]),
        "best_val_acc_epoch": hist["val_acc"].index(max(hist["val_acc"])) + 1,
        "first_epoch_train_acc_1": next((e + 1 for e, a in enumerate(hist["train_acc"]) if a == 1.0), None),
        "train_time_s": hist["train_time_s"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # table of every epoch
    rows = ["epoch,train_loss,train_acc,val_loss,val_acc"]
    for e in range(EPOCHS):
        rows.append(f"{e + 1},{hist['train_loss'][e]:.4f},{hist['train_acc'][e]:.4f},"
                    f"{hist['val_loss'][e]:.4f},{hist['val_acc'][e]:.4f}")
    (out / "epochs.csv").write_text("\n".join(rows))

    learning_curves(hist, out / "curves.png", "my CNN - overfitting (370 train images)",
                    mark_label="overfitting starts")
    print(json.dumps(summary, indent=1))
    print("saved to", out)


if __name__ == "__main__":
    main()
