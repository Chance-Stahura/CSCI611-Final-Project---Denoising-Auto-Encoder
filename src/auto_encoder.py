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

BASE_DIR: Path = Path(__file__).resolve().parent

dataset_url: str = (
    "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
)
cbsd_path: str = tf.keras.utils.get_file(
    "cbsd68.zip",
    origin=dataset_url,
    extract=True,
    cache_dir=".",  # current directory
    cache_subdir="data",  # save to ./data/
)

cbsd_img_folder: str = join(
    "data", "cbsd68_extracted", "CBSD68-dataset-master", "CBSD68", "original_png"
)


bsd500_train: Path = Path(BASE_DIR / "data/bsd500/data/images/train")
bsd500_val: Path = Path(BASE_DIR / "data/bsd500/data/images/val")

cbsd_ground_truth: Path = (
    BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/original_png"
)
cbsd_noise: Path = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/"


def build_image_set(folder: Path) -> list[str]:
    """Builds the image set"""

    image_paths: list[str] = []

    for file in listdir(folder):
        path: str = join(folder, file)
        if isfile(path):
            image_paths.append(path)
    return image_paths


if __name__ == "__main__":
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
