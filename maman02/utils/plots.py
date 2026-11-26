"""
plots.py - learning curves, confusion matrix, ROC and PR curves.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm


def learning_curves(hist, path, title=""):
    """Loss and accuracy of train and val per epoch, best epoch marked."""
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, name in ((axes[0], "loss", "loss"), (axes[1], "acc", "accuracy")):
        ax.plot(ep, hist[f"train_{key}"], label="train")
        ax.plot(ep, hist[f"val_{key}"], label="val")
        if "best_epoch" in hist:
            ax.axvline(hist["best_epoch"], color="gray", ls="--", lw=1,
                       label=f"best val loss (epoch {hist['best_epoch']})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion(cm, class_names, path, title=""):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_pr(curves, path_roc, path_pr):
    """curves: {name: (p_dog, y)}. Several models can go on the same plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (p, y) in curves.items():
        fpr, tpr, _ = skm.roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {skm.roc_auc_score(y, p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("ROC (positive = dog)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path_roc, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (p, y) in curves.items():
        prec, rec, _ = skm.precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP = {skm.average_precision_score(y, p):.4f})")
    # a random classifier gets precision = fraction of dogs
    ax.axhline(np.mean(next(iter(curves.values()))[1]), color="k", ls="--", lw=1,
               label="random")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall (positive = dog)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path_pr, dpi=150)
    plt.close(fig)
