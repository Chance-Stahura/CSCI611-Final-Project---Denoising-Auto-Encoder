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

BASE_DIR = Path(__file__).resolve().parent

#download + extract dataset
url = 'https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip'
cbsd_path = tf.keras.utils.get_file(
    'cbsd68.zip', 
    origin = url, 
    extract=True,
    cache_dir = '.', # current directory
    cache_subdir = 'data' # save to ./data/
)

#get the actual image files
#get base dir where cbsd zip is extracted, create data sub directories for data
cbsd_img_folder = os.path.join('data', 'cbsd68_extracted', 'CBSD68-dataset-master', 'CBSD68', 'original_png')


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

# 
noise_functions = {
    "gaussian": lambda x: add_gaussian_noise(x, sigma=0.2),
    "salt_pepper": lambda x: add_salt_pepper_noise(x, p=0.1),
    "occlusion": lambda x: add_occlusion(x, size=12),
}


# %%
#define paths
bsd500_train = BASE_DIR / "data/bsd500/data/images/train"
bsd500_val = BASE_DIR / "data/bsd500/data/images/val"

cbsd_ground_truth = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/original_png"
cbsd_noise = BASE_DIR / "data/cbsd68/CBSD68-dataset-master/CBSD68/"


# %% [markdown]
# Preprocess data(2)

# %%
# gather training image / validation image paths
training_imgs = []

for f in os.listdir(bsd500_train):
    path = os.path.join(bsd500_train, f)
    if os.path.isfile(path):
        training_imgs.append(path)

training_val = []

for f in os.listdir(bsd500_val):
        path = os.path.join(bsd500_val, f)
        if os.path.isfile(path):
            training_val.append(path)


# %% [markdown]
# Preprocess data(3)

# %%
# preprocess BSD500 dataset
## create dataset class
# WE ARE USING THIS CLASS BC WE ARE USING 'ON DEMAND' PATCH EXTRACTION
class dataset(tf.keras.utils.Sequence):
  
    #constructor method
    def __init__(self, image_paths, patch_size = 64, sigma = 25, batch_size = 32, training = True):
        self.img_dims = []
        self.image_paths = image_paths
        for i in self.image_paths:
            img = load_img(i)
            self.img_dims.append((img.size))
        self.patch_size = patch_size
        self.sigma = sigma / 255.0
        self.batch_size = batch_size
        self.training = training

    def __len__(self):            
        # calculate patches per image
        patches_sum = 0

        for i in self.img_dims:
            height = i[0]
            width = i[1]
            total_patches = (height // self.patch_size) * (width // self.patch_size)
            # sum them all up, divide by batch_size
            patches_sum += total_patches
        return (patches_sum // self.batch_size)

    def __getitem__(self, idx):
        # load image
        path = self.image_paths[idx]
        img_tensor = load_img(path)
        # convert to tensor: (N, H, W, C)
        img_tensor = tf.keras.utils.img_to_array(img_tensor)
        img_tensor = img_tensor / 255.0
        img_tensor = tf.expand_dims(img_tensor, axis = 0)
        # extract patches
        #RETURNS ALL FLATTENED PATCHES TOGETHER
        #DURING TRAINING THE MODEL WILL SEE ALL PATCHES FROM AN IMAGE AT ONCE
        clean_patch = tf.image.extract_patches(
            images = img_tensor,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID'
            )
        # add noise
        #tf.random.uniform()?
        added_noise = tf.random.normal(shape = tf.shape(clean_patch), stddev = self.sigma)
        noisy_patch = tf.add(clean_patch, added_noise)
        noisy_patch = tf.clip_by_value(noisy_patch, 0, 1)

        return (noisy_patch, clean_patch)


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
