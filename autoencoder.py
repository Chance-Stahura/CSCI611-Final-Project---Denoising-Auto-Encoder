# %% [markdown]
# Import libraries and load the dataset

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

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
# Preprocess data
# normalize pixel values 

# %%
    # create dataset class
    # apply transformations
    # add noise on the fly
    # split data
    # create data loaders


# %% [markdown]
# Add noise to images
# Gaussian blur, poisson, etc
# use original as the reconstruction targets

# %%


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

# %% [markdown]
# Configure training

# %%



# %% [markdown]
# Train the model

# %%



# %% [markdown]
# Evaluate and test

# %%