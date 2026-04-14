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

BASE_DIR: Path = Path(__file__).resolve().parent.parent

cbsd68_url: str = (
    "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
)
cbsd68_path: str = tf.keras.utils.get_file(
    "cbsd68.zip",
    origin=cbsd68_url,
    extract=True,
    cache_dir=".",  # current directory
    cache_subdir="data",  # save to ./data/
)

cbsd68_img_folder: str = join(
    "data", "cbsd68_extracted", "CBSD68-dataset-master", "CBSD68", "original_png"
)

bsds500_url: str = (
    "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
)
bsds500_path: str = tf.keras.utils.get_file(
    "BSDS500.zip",
    origin=bsds500_url,
    extract=True,
    cache_dir=".",  # current directory
    cache_subdir="data",  # save to ./data/
)
# get the actual image files
# get base dir where cbsd zip is extracted, create data sub directories for data
bsds500_img_folder: str = join(
    "data", "BSDS500_extracted", "BSDS500-master", "BSDS500", "data", "images"
)


bsd500_train: Path = Path(BASE_DIR / "data/bsd500/data/images/train")
bsd500_val: Path = Path(BASE_DIR / "data/bsd500/data/images/val")

cbsd_ground_truth: Path = (
    BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/original_png"
)
cbsd_noise: Path = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/"


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
