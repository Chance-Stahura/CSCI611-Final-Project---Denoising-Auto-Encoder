"""This is a based python code from tensorflow's autoencoder example."""

# https://www.tensorflow.org/tutorials/generative/autoencoder

# coding: utf-8

from os import listdir
from os.path import isfile, join
from pathlib import Path

import tensorflow as tf
from keras import layers, models

from dataset import Dataset
from download_dataset import (
    download_dataset,
    TARGET_DIR_BSDS500,
    TARGET_DIR_CBSD68,
)

PATCH_SIZE: int = 64
NOISE_SIGMA: int = 25

TRAIN_BATCH_SIZE: int = 32
VALIDATION_BATCH_SIZE: int = 32
TEST_BATCH_SIZE: int = 1

LEARNING_RATE: float = 1e-3
EPOCHS: int = 3

IMAGE_CHANNELS: int = 3


BASE_DIR: Path = Path(__file__).resolve().parents[1]
SAVE_DIR: Path = BASE_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)
MODEL_SAVE_PATH: Path = SAVE_DIR / "original_benchmark.keras"

cbsd68_img_folder: Path = download_dataset(TARGET_DIR_CBSD68)
cbsd_ground_truth: Path = cbsd68_img_folder / "original_png"

bsds500_img_folder: Path = download_dataset(TARGET_DIR_BSDS500)
bsd500_train: Path = bsds500_img_folder / "train"
bsd500_val: Path = bsds500_img_folder / "val"
bsd500_test: Path = bsds500_img_folder / "test"


def build_image_set(folder: Path) -> list[str]:
    """Builds a sorted image path list."""
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


def main() -> None:
    """The main code."""
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
    test_imgs: list[str] = build_image_set(cbsd_ground_truth)

    train_ds: Dataset = Dataset(
        image_paths=training_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TRAIN_BATCH_SIZE,
        training=True,
        return_full_image=False,
        shuffle=True,
        noise_type="gaussian",
    )

    val_ds: Dataset = Dataset(
        image_paths=validation_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=VALIDATION_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
        noise_type="gaussian",
    )

    test_patch_ds: Dataset = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
        noise_type="gaussian",
    )

    test_full_ds: Dataset = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=NOISE_SIGMA,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=True,
        shuffle=False,
        noise_type="gaussian",
        pad_multiple=4,
    )

    train_model: tf.keras.Model = build_original_tf_benchmark_model(
        input_shape=(PATCH_SIZE, PATCH_SIZE, IMAGE_CHANNELS)
    )

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    train_model.summary()

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    train_model.save(MODEL_SAVE_PATH)
    print(f"Saved model to: {MODEL_SAVE_PATH}")

    test_model: tf.keras.Model = build_original_tf_benchmark_model(
        input_shape=(None, None, IMAGE_CHANNELS)
    )
    test_model.set_weights(train_model.get_weights())
    test_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    patch_test_results = train_model.evaluate(test_patch_ds)
    print(f"Patch test results: {patch_test_results}")

    full_avg_mse, full_avg_mae = evaluate_full_image_dataset(test_model, test_full_ds)
    print(f"Full-image test results: [{full_avg_mse}, {full_avg_mae}]")


if __name__ == "__main__":
    main()
