# %% [markdown]
# Import libraries and load the dataset

# %%

# import numpy as np
from numpy import ndarray

# import matplotlib.pyplot as plt

import tensorflow as tf
import os
from pathlib import Path

# from PIL import Image
from PIL.Image import Image as PILImage

# from keras import layers, models, optimizers
from keras.utils import load_img

# from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

BASE_DIR: Path = Path(__file__).resolve().parent

# download + extract dataset
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

# get the actual image files
# get base dir where cbsd zip is extracted, create data sub directories for data
cbsd_img_folder: str = os.path.join(
    "data", "cbsd68_extracted", "CBSD68-dataset-master", "CBSD68", "original_png"
)


# %% [markdown]
# Preprocess data(1)

# %%
# define paths
bsd500_train: Path = Path(BASE_DIR / "data/bsd500/data/images/train")
bsd500_val: Path = Path(BASE_DIR / "data/bsd500/data/images/val")

cbsd_ground_truth: Path = (
    BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/original_png"
)
cbsd_noise: Path = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/"


# %% [markdown]
# Preprocess data(2)

# %%
# gather training image / validation image paths


def build_image_set(folder: Path) -> list[str]:
    """Builds the image set"""

    image_paths: list[str] = []

    for f in os.listdir(folder):
        path: str = os.path.join(folder, f)
        if os.path.isfile(path):
            image_paths.append(path)
    return image_paths


# %% [markdown]
# Preprocess data(3)


# %% [markdown]
# Define noise models:
# Gaussian noise
def add_gaussian_noise(x, sigma=0.2):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=sigma)
    x_noisy = x + noise
    return tf.clip_by_value(x_noisy, 0.0, 1.0)


# salt-and-pepper noise
def add_salt_pepper_noise(x, p=0.1):
    random_vals = tf.random.uniform(tf.shape(x))

    salt = tf.cast(random_vals > (1 - p / 2), tf.float32)
    pepper = tf.cast(random_vals < (p / 2), tf.float32)

    x_noisy = x * (1 - salt - pepper) + salt
    return x_noisy


# structured noise (random occlusion)
def add_occlusion(x, size=12):
    h, w = tf.shape(x)[0], tf.shape(x)[1]

    top = tf.random.uniform([], 0, h - size, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - size, dtype=tf.int32)

    mask = tf.ones_like(x)

    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.reshape(
            tf.stack(
                tf.meshgrid(
                    tf.range(top, top + size),
                    tf.range(left, left + size),
                    indexing="ij",
                ),
                axis=-1,
            ),
            [-1, 2],
        ),
        updates=tf.zeros([size * size]),
    )

    return x * mask


# Noise registry
def get_noise_fn(config):
    noise_type = config["noise"]["type"]

    if noise_type == "gaussian":
        sigma = config["noise"]["sigma"]
        return lambda x: add_gaussian_noise(x, sigma)

    elif noise_type == "salt_pepper":
        p = config["noise"]["p"]
        return lambda x: add_salt_pepper_noise(x, p)

    elif noise_type == "occlusion":
        size = config["noise"]["size"]
        return lambda x: add_occlusion(x, size)

    else:
        raise ValueError("Unknown noise type")


# %%
# preprocess BSD500 dataset
# create dataset class
# WE ARE USING THIS CLASS BC WE ARE USING 'ON DEMAND' PATCH EXTRACTION
class Dataset(tf.keras.utils.Sequence):

    def __init__(
        self,
        image_paths: list[str],
        patch_size: int = 64,
        sigma: int = 25,
        batch_size: int = 32,
        training: bool = True,
    ) -> None:
        """Dataset Constructor"""
        # self.img_dims = []
        self.image_paths = image_paths
        # for i in self.image_paths:
        #     img: PILImage = load_img(i)
        #     self.img_dims.append((img.size))
        self.patch_size: int = patch_size
        self.sigma: float = sigma / 255.0
        self.batch_size: int = batch_size
        self.training: bool = training

    def __len__(self) -> int:
        return len(self.image_paths)

    # def old__len__(self) -> int:
    #     """DEPRECATED, possibly a mistake..."""
    #     """Returns the number of batches per epoch"""
    #     patches_sum: int = 0

    #     # sum the each image's patches
    #     for i in self.img_dims:
    #         width: int = i[0]
    #         height: int = i[1]
    #         total_patches: int = (height // self.patch_size) * (
    #             width // self.patch_size
    #         )
    #         patches_sum += total_patches

    #     return patches_sum // self.batch_size

    def _load_image_as_tensor(self, path: str) -> tf.Tensor:
        """Loads an image as a tensor"""
        img_tensor: PILImage = load_img(path)
        # convert to tensor: (N, H, W, C)
        tensor_array: ndarray = tf.keras.utils.img_to_array(img_tensor)
        tensor_array = tensor_array / 255.0
        return tf.expand_dims(tensor_array, axis=0)

    def _extract_patches(self, img_tensor: tf.Tensor) -> tf.Tensor:
        """Extracts patches from an image"""
        return tf.image.extract_patches(
            images=img_tensor,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

    def _add_noise(self, clean_patch: tf.Tensor) -> tf.Tensor:
        """Adds noise to a clean patch"""
        added_noise: tf.Tensor = tf.random.normal(
            shape=tf.shape(clean_patch), stddev=self.sigma
        )
        noisy_patch: tf.Tensor = tf.add(clean_patch, added_noise)
        return tf.clip_by_value(noisy_patch, 0, 1)

    def __getitem__(self, idx: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns a batch of noisy and clean patches."""
        # load image
        path: str = self.image_paths[idx]
        img_tensor: tf.Tensor = self._load_image_as_tensor(path)
        clean_patch: tf.Tensor = self._extract_patches(img_tensor)
        # extract patches
        # RETURNS ALL FLATTENED PATCHES TOGETHER
        # DURING TRAINING THE MODEL WILL SEE ALL PATCHES FROM AN IMAGE AT ONCE
        # add noise
        # tf.random.uniform()?
        noisy_patch: tf.Tensor = self._add_noise(clean_patch)
        return (noisy_patch, clean_patch)


# %% [markdown]
# Preprocess data(4)

# %%
# define CBSD68 test pairing logic
# create data loaders / piping logic


# %% [markdown]
# Build autoencoder model

# %%

# input
# encode
# conv
# relu
# conv
# conv
# dense
# latent space
# dense
# decode
# dense
# dense
# conv
# conv
# relu
# conv

if __name__ == "__main__":
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
