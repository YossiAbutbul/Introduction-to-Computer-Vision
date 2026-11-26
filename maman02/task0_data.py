"""
task0_data.py - preparing the data for all tasks.
Downloads the dataset, makes the 80/20 train/val split by breed
and saves the image counts and a few example images.
"""

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import config
from utils.data import (breed_names, count, download, get_split, get_transforms,
                        read_split)


def main():
    out = config.OUT_ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)

    download()
    train, val = get_split()
    test = read_split("test")
    names = breed_names(train + val)

    # counts per breed and per species
    lines = [f"train {len(train)}  val {len(val)}  test {len(test)}", ""]
    for split_name, rs in (("train", train), ("val", val), ("test", test)):
        sp = count(rs, 2)
        lines.append(f"{split_name}: cats {sp[0]}  dogs {sp[1]}  "
                     f"dog fraction {sp[1] / len(rs):.3f}")
    lines += ["", f"{'breed':28s} train  val  test  val%"]
    c_tr, c_va, c_te = count(train, 1), count(val, 1), count(test, 1)
    for b, n in enumerate(names):
        lines.append(f"{n:28s} {c_tr[b]:5d} {c_va[b]:4d} {c_te[b]:5d}  "
                     f"{100 * c_va[b] / (c_tr[b] + c_va[b]):.1f}")
    text = "\n".join(lines)
    print(text)
    (out / "split_summary.txt").write_text(text)
    # copy of the split file next to the results
    shutil.copy(config.SPLIT_FILE, out / config.SPLIT_FILE.name)

    # bar plot of the counts
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x, [c_tr[b] for b in x], label="train")
    ax.bar(x, [c_va[b] for b in x], bottom=[c_tr[b] for b in x], label="val")
    ax.set_xticks(x, names, rotation=90, fontsize=8)
    ax.set_ylabel("images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "breed_counts.png", dpi=150)
    plt.close(fig)

    # 6 train images: original + 2 random augmentations of each
    aug = get_transforms(224, augment=True)
    rng = np.random.default_rng(0)
    picks = [train[i] for i in rng.choice(len(train), 6, replace=False)]
    img_dir = config.DATA_ROOT / "oxford-iiit-pet" / "images"
    fig, axes = plt.subplots(3, 6, figsize=(14, 7.5))
    for j, (name, breed, species) in enumerate(picks):
        img = Image.open(img_dir / f"{name}.jpg").convert("RGB")
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{names[breed]}\n({'dog' if species else 'cat'})",
                             fontsize=8)
        for k in (1, 2):
            t = aug(img).permute(1, 2, 0).numpy() * config.STD + config.MEAN
            axes[k, j].imshow(np.clip(t, 0, 1))
    for a in axes.flat:
        a.axis("off")
    axes[0, 0].text(-0.1, 0.5, "original", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "augmented", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=120)
    plt.close(fig)
    print("saved to", out)


if __name__ == "__main__":
    main()
