"""
data.py - loading Oxford-IIIT Pet, the train/val split and the data loaders.

Labels I use for every image:
  breed   0..36
  species 0 = cat, 1 = dog
"""

import json
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from . import config


def download(root=config.DATA_ROOT):
    """Download the dataset with torchvision (skipped if already there)."""
    for split in ("trainval", "test"):
        datasets.OxfordIIITPet(root, split=split, download=True)


def read_split(split, root=config.DATA_ROOT):
    """Read annotations/<split>.txt -> list of (name, breed, species).
    A line looks like: 'Abyssinian_100 1 1 1'."""
    path = root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
    records = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, class_id, species, _ = line.split()
            # the file counts from 1, I want labels from 0
            records.append((name, int(class_id) - 1, int(species) - 1))
    return records


def breed_names(records):
    """Breed name for each class id, taken from the file name."""
    names = {}
    for name, breed, _ in records:
        names[breed] = name.rsplit("_", 1)[0]
    return [names[i] for i in range(len(names))]


def stratified_split(records, val_fraction=config.VAL_FRACTION, seed=config.SPLIT_SEED):
    """Shuffle every breed separately and move val_fraction of it to validation."""
    rng = np.random.default_rng(seed)
    by_breed = {}
    for r in records:
        by_breed.setdefault(r[1], []).append(r[0])

    train, val = [], []
    for breed in sorted(by_breed):
        names = sorted(by_breed[breed])
        names = [names[i] for i in rng.permutation(len(names))]
        n_val = round(len(names) * val_fraction)
        val += names[:n_val]
        train += names[n_val:]
    return sorted(train), sorted(val)


def get_split(root=config.DATA_ROOT, split_file=config.SPLIT_FILE):
    """Train and val records. The split is made once and saved to a file,
    after that it is only loaded, so all tasks use the same images."""
    records = read_split("trainval", root)
    if split_file.exists():
        saved = json.loads(split_file.read_text())
        train_names, val_names = saved["train"], saved["val"]
        assert set(train_names) | set(val_names) == {r[0] for r in records}, \
            "split file does not match the dataset"
    else:
        train_names, val_names = stratified_split(records)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps({
            "val_fraction": config.VAL_FRACTION, "seed": config.SPLIT_SEED,
            "train": train_names, "val": val_names}, indent=0))

    train_set, val_set = set(train_names), set(val_names)
    train = [r for r in records if r[0] in train_set]
    val = [r for r in records if r[0] in val_set]
    return train, val


def subset_per_breed(records, n_per_breed, seed=0):
    """A small training set: n random images from every breed."""
    rng = np.random.default_rng(seed)
    out = []
    for breed in sorted({r[1] for r in records}):
        rs = [r for r in records if r[1] == breed]
        idx = rng.permutation(len(rs))[:n_per_breed]
        out += [rs[i] for i in sorted(idx)]
    return out


def get_transforms(img_size, augment):
    """With augment: random crop, flip and small color change (train only).
    Without: resize and center crop."""
    norm = transforms.Normalize(config.MEAN, config.STD)
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), norm])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm])


class PetsDataset(Dataset):
    """Returns (image, label). target is 'species' or 'breed'."""

    def __init__(self, records, target, transform, root=config.DATA_ROOT):
        self.records = records
        self.col = {"breed": 1, "species": 2}[target]
        self.transform = transform
        self.img_dir = root / "oxford-iiit-pet" / "images"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        # some images are grayscale or RGBA
        img = Image.open(self.img_dir / f"{r[0]}.jpg").convert("RGB")
        return self.transform(img), r[self.col]

    def labels(self):
        return [r[self.col] for r in self.records]


def make_loaders(target, img_size=224, batch_size=64, augment=True,
                 train_records=None, num_workers=2, root=config.DATA_ROOT):
    """DataLoaders for train, val and test.
    train_records can replace the train set (for a smaller subset)."""
    train, val = get_split(root)
    if train_records is not None:
        train = train_records
    test = read_split("test", root)

    eval_tf = get_transforms(img_size, augment=False)
    sets = {
        "train": PetsDataset(train, target, get_transforms(img_size, augment), root),
        "val": PetsDataset(val, target, eval_tf, root),
        "test": PetsDataset(test, target, eval_tf, root),
    }
    pin = torch.cuda.is_available()
    return {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train"),
                          num_workers=num_workers, pin_memory=pin,
                          persistent_workers=num_workers > 0)
            for k, ds in sets.items()}


def count(records, col):
    """How many records have each value in column col."""
    return Counter(r[col] for r in records)
