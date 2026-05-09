"""This is a based python code from tensorflow's autoencoder example."""

# https://www.tensorflow.org/tutorials/generative/autoencoder

# coding: utf-8

import tensorflow as tf
from keras import layers, models


IMAGE_CHANNELS: int = 3


def build_original_tf_benchmark_model(
    input_shape: tuple[
        int | None,
        int | None,
        int,
    ],
) -> tf.keras.Model:
    """This will build the auto_encoder model."""

    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)   # conv 1
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)       # conv 2
    x = layers.MaxPooling2D(2, padding="same")(x)                         # pool -> 32x32x128

    # Decoder
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)  # mirror conv 2 + upsample -> 64x64x128
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)              # mirror conv 1, no stride -> 64x64x64

    outputs = layers.Conv2D(IMAGE_CHANNELS, 3, activation="sigmoid", padding="same")(x)

    return models.Model(
        inputs=inputs, outputs=outputs, name="original_tf_benchmark_model"
    )
