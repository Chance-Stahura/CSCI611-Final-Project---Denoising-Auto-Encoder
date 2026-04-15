# import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from os import listdir
from os.path import isfile, join
from pathlib import Path

# from keras import layers, models, optimizers

# from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

# from dataset import Dataset
# from noise import add_noise

from scripts.download_dataset import get_cbsd68_path
from scripts.download_dataset import get_bsds500_path

BASE_DIR: Path = Path(__file__).resolve().parents[2]

cbsd68_img_folder: Path = get_cbsd68_path()
cbsd_ground_truth: Path = cbsd68_img_folder / "original_png"

bsds500_img_folder: Path = get_bsds500_path()
bsd500_train: Path = bsds500_img_folder / "train"
bsd500_val: Path = bsds500_img_folder / "val"
bsd500_test: Path = bsds500_img_folder / "test"


def build_image_set(folder: Path) -> list[str]:
    """Builds the image set"""

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    image_paths: list[str] = []

    for file in listdir(folder):
        path: str = join(folder, file)
        if isfile(path):
            image_paths.append(path)
    return image_paths


if __name__ == "__main__":
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
