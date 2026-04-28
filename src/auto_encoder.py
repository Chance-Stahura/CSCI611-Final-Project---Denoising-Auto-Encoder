"""This is the main code for the autoencoder."""

# coding: utf-8

from os import listdir
from os.path import isfile, join
from pathlib import Path

import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
import json

from dataset import Dataset
# to train all three models at one time
from original_benchmark import build_original_tf_benchmark_model

from download_dataset import (
    download_dataset,
    TARGET_DIR_BSDS500,
    TARGET_DIR_CBSD68,
)

PATCH_SIZE: int = 64
NOISE_SIGMA: int = 25
TRAIN_BATCH_SIZE: int = 32
VAL_BATCH_SIZE: int = 32
TEST_BATCH_SIZE: int = 1

LEARNING_RATE: float = 1e-3
EPOCHS: int = 3  # change back to 20

INPUT_CHANNELS: int = 3
FILTERS_STAGE_1: int = 64
FILTERS_STAGE_2: int = 128
CONV_KERNEL_SIZE: int = 3
POOL_FACTOR: int = 2
UPSAMPLE_FACTOR: int = 2

TRAIN_INPUT_SHAPE: tuple[int, int, int] = (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS)
FULL_IMAGE_INPUT_SHAPE: tuple[None, None, int] = (None, None, INPUT_CHANNELS)


BASE_DIR: Path = Path(__file__).resolve().parents[1]
SAVE_DIR: Path = BASE_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)


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


# Got help from ChatGPT on this one.
def evaluate_full_image_dataset(
    model: tf.keras.Model,
    dataset: Dataset,
) -> tuple[float, float]:
    """Evaluates a full-image dataset one sample at a time."""
    total_mse: float = 0.0
    total_mae: float = 0.0
    num_batches: int = len(dataset)

    for i in range(num_batches):
        noisy_batch, clean_batch = dataset[i]

        predictions: tf.Tensor = model.predict(noisy_batch, verbose=0)

        mse: float = tf.reduce_mean(tf.square(clean_batch - predictions)).numpy().item()
        mae: float = tf.reduce_mean(tf.abs(clean_batch - predictions)).numpy().item()

        total_mse += mse
        total_mae += mae

    avg_mse: float = total_mse / num_batches
    avg_mae: float = total_mae / num_batches

    return avg_mse, avg_mae


# This code is referenced from:
# Omar Hankare: Autoencoders explained
# Link: https://ompramod.medium.com/autoencoders-explained-9196c38af6f6

# NOTE: this code was adapted for the purpose of this project


def build_dense_model(
    input_shape: tuple[
        int | None,
        int | None,
        int,
    ] = TRAIN_INPUT_SHAPE,
) -> tf.keras.Model:
    """This will build the fully connected model."""
    # (64 x 64) x 3 = 12288
    inputs = layers.Input(shape=input_shape)

    x = layers.Flatten()(inputs)

    # encoder
    encoded = layers.Dense(128, activation="relu")(x)
    encoded = layers.Dense(64, activation="relu")(encoded)
    encoded = layers.Dense(32, activation="relu")(encoded)

    # decoder
    decoded = layers.Dense(64, activation="relu")(encoded)
    decoded = layers.Dense(128, activation="relu")(decoded)
    decoded = layers.Dense(12288, activation="sigmoid")(decoded)

    output = layers.Reshape(input_shape)(decoded)

    return models.Model(inputs, output, name="dense_autoencoder")


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

    # Have 2 test datasets for patch and full image

    test_patch_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
    )

    test_full_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=True,
        shuffle=False,
        pad_multiple=2,
    )

    # this can work with multiple models
    models_to_run = {
        "denoising_autoencoder": build_autoencoder(),
        "dense_autoencoder": build_dense_model(input_shape=TRAIN_INPUT_SHAPE),
        "original_benchmark": build_original_tf_benchmark_model(input_shape=TRAIN_INPUT_SHAPE),
    }

    for (
        name,
        model,
    ) in models_to_run.items():
        print(f"\n{'=' * 40}")
        print(f" Running model: {name}")
        print(f"{'=' * 40}\n")

        # model: tf.keras.Model = build_autoencoder(input_shape=TRAIN_INPUT_SHAPE)

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

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )

        histories_path: Path = BASE_DIR / "histories"
        histories_path.mkdir(parents=True, exist_ok=True)

        # saves the history of each model for use in evaluate.py
        # for plotting training/validation losses
        with open(f"histories/{name}_history.json", "w") as f:
            json.dump(history.history, f)

        model_save_path: Path = SAVE_DIR / f"{name}.keras"
        model.save(model_save_path, overwrite=True)
        print(f"Saved model [{name}] to: {model_save_path}")

        if name == "denoising_autoencoder":
            full_image_model: tf.keras.Model = build_autoencoder(
                input_shape=FULL_IMAGE_INPUT_SHAPE
            )
            full_image_model.set_weights(model.get_weights())

            full_model_save_path: Path = SAVE_DIR / f"{name}_full_image.keras"
            full_image_model.save(full_model_save_path)
            print(f"Saved full-image model [{name}] to: {full_model_save_path}")

            full_image_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="mse",  # mean squared error
                metrics=["mae"],  # mean absolute error
            )

            avg_mse, avg_mae = evaluate_full_image_dataset(
                full_image_model, test_full_ds
            )
            print(f"Test results [{name}]: [{avg_mse}, {avg_mae}]")
        else:
            test_results = model.evaluate(test_patch_ds)
            print(f"Test results [{name}]: {test_results}")


if __name__ == "__main__":
    main()
