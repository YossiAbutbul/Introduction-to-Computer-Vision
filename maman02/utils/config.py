"""
config.py - paths and parameters used by all tasks.
"""

import os
from pathlib import Path

MAMAN_DIR = Path(__file__).resolve().parent.parent

# where the dataset is (on Colab: /content/data)
DATA_ROOT = Path(os.environ.get("PETS_ROOT", MAMAN_DIR / "data"))

# where results are saved (on Colab: a folder in Google Drive)
OUT_ROOT = Path(os.environ.get("OUT_ROOT", MAMAN_DIR / "outputs"))

# train/val split: 20% of every breed goes to validation
VAL_FRACTION = 0.2
SPLIT_SEED = 0
SPLIT_FILE = MAMAN_DIR / "splits" / "trainval_split.json"

N_BREEDS = 37

# ImageNet mean and std for normalizing the images
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
