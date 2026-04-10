# %% [markdown]
# Import libraries and load the dataset

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from pathlib import Path
from PIL import Image
from keras import layers, models, optimizers
from keras.utils import load_img
from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose
from scripts.download_dataset import download_cbsd68

BASE_DIR = Path(__file__).resolve().parent

#download + extract dataset
#get the actual image files
#get base dir where cbsd zip is extracted, create data sub directories for data
cbsd68_img_folder = download_cbsd68(BASE_DIR)


# %% [markdown]
# Preprocess data(1)

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

    salt = tf.cast(random_vals > (1 - p/2), tf.float32)
    pepper = tf.cast(random_vals < (p/2), tf.float32)

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
            tf.stack(tf.meshgrid(
                tf.range(top, top + size),
                tf.range(left, left + size),
                indexing='ij'
            ), axis=-1),
            [-1, 2]
        ),
        updates=tf.zeros([size * size])
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
# example usage:
# noise_fn = get_noise_fn(config)

# %%
#define paths
bsd500_train = BASE_DIR / "data/bsd500/data/images/train"
bsd500_val = BASE_DIR / "data/bsd500/data/images/val"

cbsd_ground_truth = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/original_png"
cbsd_noise = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/"


# %% [markdown]
# Preprocess data(4)

# %%
# define CBSD68 test pairing logic
#create data loaders / piping logic


# %% [markdown]
# Build autoencoder model

# %%

#input
#encode
    #conv
    #relu
    #conv
    #conv
    #dense
#latent space
    #dense
#decode
    #dense
    #dense
    #conv
    #conv
    #relu
    #conv
