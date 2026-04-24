"""This is the main code for the autoencoder."""

# coding: utf-8

from os import listdir
from os.path import isfile, join
from pathlib import Path

from keras import layers, Model, optimizers, Sequential

from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

from keras.constraints import max_norm

from dataset import Dataset
# from noise import add_noise

import tensorflow as tf
from keras import layers, models

from .download_dataset import (
    download_dataset,
    TARGET_DIR_BSDS500,
    TARGET_DIR_CBSD68,
)
from .dataset import Dataset

PATCH_SIZE: int = 64
NOISE_SIGMA: int = 25
TRAIN_BATCH_SIZE: int = 32
VAL_BATCH_SIZE: int = 32
TEST_BATCH_SIZE: int = 1

LEARNING_RATE: float = 1e-3
EPOCHS: int = 20

INPUT_CHANNELS: int = 3
FILTERS_STAGE_1: int = 64
FILTERS_STAGE_2: int = 128
CONV_KERNEL_SIZE: int = 3
POOL_FACTOR: int = 2
UPSAMPLE_FACTOR: int = 2

TRAIN_INPUT_SHAPE: tuple[int, int, int] = (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS)
FULL_IMAGE_INPUT_SHAPE: tuple[None, None, int] = (None, None, INPUT_CHANNELS)


BASE_DIR: Path = Path(__file__).resolve().parents[2]

cbsd68_img_folder: Path = download_dataset(TARGET_DIR_CBSD68)
cbsd_ground_truth: Path = cbsd68_img_folder / "original_png"

bsds500_img_folder: Path = download_dataset(TARGET_DIR_BSDS500)
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

    return sorted(image_paths)


def build_autoencoder(
    input_shape: tuple[
        int | None,
        int | None,
        int,
    ] = FULL_IMAGE_INPUT_SHAPE,
) -> tf.keras.Model:
    """This will build the auto_encoder model."""
    inputs = layers.Input(shape=input_shape)

    x: tf.Tensor = layers.Conv2D(
        FILTERS_STAGE_1, CONV_KERNEL_SIZE, activation="relu", padding="same"
    )(inputs)
    x = layers.Conv2D(
        FILTERS_STAGE_1, CONV_KERNEL_SIZE, activation="relu", padding="same"
    )(x)
    x = layers.MaxPooling2D(POOL_FACTOR, padding="same")(x)

    x = layers.Conv2D(
        FILTERS_STAGE_2, CONV_KERNEL_SIZE, activation="relu", padding="same"
    )(x)
    x = layers.Conv2D(
        FILTERS_STAGE_2, CONV_KERNEL_SIZE, activation="relu", padding="same"
    )(x)
    x = layers.UpSampling2D(UPSAMPLE_FACTOR)(x)

    outputs = layers.Conv2D(
        INPUT_CHANNELS, CONV_KERNEL_SIZE, activation="sigmoid", padding="same"
    )(x)

    return models.Model(inputs, outputs, name="denoising_autoencoder")


def main() -> None:
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
    test_imgs: list[str] = build_image_set(cbsd_ground_truth)

    train_ds = Dataset(
        image_paths=training_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TRAIN_BATCH_SIZE,
        training=True,
        return_full_image=False,
        shuffle=True,
    )

    val_ds = Dataset(
        image_paths=validation_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=VAL_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
    )

    test_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
    )

    model: tf.keras.Model = build_autoencoder(input_shape=TRAIN_INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",  # mean squared error
        metrics=["mae"],  # mean absolute error
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    full_image_model: tf.keras.Model = build_autoencoder(
        input_shape=FULL_IMAGE_INPUT_SHAPE
    )
    full_image_model.set_weights(model.get_weights())

    full_image_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",  # mean squared error
        metrics=["mae"],  # mean absolute error
    )

    test_results = full_image_model.evaluate(test_ds)
    print(f"Test results: {test_results}")

    return


if __name__ == "__main__":
    main()
