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

    inputs = layers.Input(shape=input_shape) # 64x64x3

  # Encoder
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)            # 64x64 -> 32x32
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)            # 32x32 -> 16x16
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)            # 16x16 -> 8x8x128 = 8,192
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)  # 8x8 -> 16x16
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)   # 16x16 -> 32x32
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)   # 32x32 -> 64x64

    outputs = layers.Conv2D(IMAGE_CHANNELS, 3, activation="sigmoid", padding="same")(x)

    return models.Model(
        inputs=inputs, outputs=outputs, name="original_tf_benchmark_model"
    )
