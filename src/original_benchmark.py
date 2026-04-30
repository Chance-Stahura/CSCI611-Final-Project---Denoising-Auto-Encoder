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
    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        activation="relu",
        padding="same",
    )(inputs)

    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(
        filters=8,
        kernel_size=3,
        activation="relu",
        padding="same",
    )(x)

    x = layers.MaxPooling2D(pool_size=2, padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(
        filters=8,
        kernel_size=3,
        strides=2,
        activation="relu",
        padding="same",
    )(x)
    x = layers.Conv2DTranspose(
        filters=16,
        kernel_size=3,
        strides=2,
        activation="relu",
        padding="same",
    )(x)

    outputs = layers.Conv2D(
        filters=IMAGE_CHANNELS,
        kernel_size=3,
        activation="sigmoid",
        padding="same",
    )(x)

    return models.Model(
        inputs=inputs, outputs=outputs, name="original_tf_benchmark_model"
    )
