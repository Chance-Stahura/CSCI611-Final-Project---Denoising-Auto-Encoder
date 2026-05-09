"""This is the main code for the autoencoder."""

# coding: utf-8

from os import listdir
from os.path import isfile, join
from pathlib import Path
import json

import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
# to train all three models at one time
from original_benchmark import build_original_tf_benchmark_model

from dataset import (
    Dataset,
    DEFAULT_SALT_PEPPER_P,
    DEFAULT_OCCLUSION_SIZE,
)

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
EPOCHS: int = 6  # change back to 20

INPUT_CHANNELS: int = 3
FILTERS_STAGE_1: int = 64
FILTERS_STAGE_2: int = 128
CONV_KERNEL_SIZE: int = 3
POOL_FACTOR: int = 2
UPSAMPLE_FACTOR: int = 2

TRAIN_INPUT_SHAPE: tuple[int, int, int] = (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNELS)
FULL_IMAGE_INPUT_SHAPE: tuple[None, None, int] = (None, None, INPUT_CHANNELS)

BASE_DIR: Path = Path(__file__).resolve().parents[1]

MODELS_DIR: Path = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DENOISE_DIR: Path = MODELS_DIR / "denoise"
DENOISE_DIR.mkdir(parents=True, exist_ok=True)

DENOISE_FULL_DIR: Path = MODELS_DIR / "denoise_full"
DENOISE_FULL_DIR.mkdir(parents=True, exist_ok=True)

DENSE_DIR: Path = MODELS_DIR / "dense"
DENSE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR: Path = MODELS_DIR / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

(DENOISE_DIR / "histories").mkdir(parents=True, exist_ok=True)
(DENOISE_FULL_DIR / "histories").mkdir(parents=True, exist_ok=True)
(DENSE_DIR / "histories").mkdir(parents=True, exist_ok=True)
(BENCHMARK_DIR / "histories").mkdir(parents=True, exist_ok=True)

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


def get_model_output_dir(model_key: str) -> Path:
    """Returns the output directory for a given model key."""
    if model_key == "denoising_autoencoder":
        return DENOISE_DIR
    if model_key == "denoising_autoencoder_full":
        return DENOISE_FULL_DIR
    if model_key == "dense_autoencoder":
        return DENSE_DIR
    if model_key == "original_benchmark":
        return BENCHMARK_DIR

    raise ValueError(f"Unknown model key: {model_key}")


def model_process(
    experiment_name: str = "default",
    noise_type: str = "gaussian",
    sigma: int = NOISE_SIGMA,
    epochs: int = EPOCHS,
    dataset: str = "cbsd68",
    salt_pepper_p: float = DEFAULT_SALT_PEPPER_P,
    occlusion_size: int = DEFAULT_OCCLUSION_SIZE,
) -> None:
    """This will build the auto_encoder model."""
    tf.random.set_seed(42)

    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)
    test_imgs: list[str] = build_image_set(cbsd_ground_truth)

    train_ds = Dataset(
        image_paths=training_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        noise_type=noise_type,
        batch_size=TRAIN_BATCH_SIZE,
        training=True,
        return_full_image=False,
        shuffle=True,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    val_ds = Dataset(
        image_paths=validation_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        noise_type=noise_type,
        batch_size=VAL_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    # Have 2 test datasets for patch and full image

    test_patch_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        noise_type=noise_type,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    test_full_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        noise_type=noise_type,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=True,
        shuffle=False,
        pad_multiple=2,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    # this can work with multiple models
    models_to_run = {
        "denoising_autoencoder": build_autoencoder(),
        "dense_autoencoder": build_dense_model(input_shape=TRAIN_INPUT_SHAPE),
        "original_benchmark": build_original_tf_benchmark_model(
            input_shape=TRAIN_INPUT_SHAPE
        ),
    }

    for (
        name,
        model,
    ) in models_to_run.items():

        print(f"\n{'=' * 40}")
        print(f" Running model: {name}_{experiment_name}")
        print(f"{'=' * 40}\n")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="mse",  # mean squared error
            metrics=["mae"],  # mean absolute error
        )

        model.summary()

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        model_dir: Path = get_model_output_dir(name)
        model_save_path: Path = model_dir / f"{experiment_name}.keras"
        model.save(model_save_path, overwrite=True)
        print(f"Saved model [{name}_{experiment_name}] to: {model_save_path}")

        # saves the history of each model for use in evaluate.py
        # for plotting training/validation losses
        history_dir: Path = model_dir / "histories"
        history_path: Path = history_dir / f"{experiment_name}_history.json"

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history.history, f)

        if name != "denoising_autoencoder":
            test_results = model.evaluate(test_patch_ds)
            print(f"Test results [{name}_{experiment_name}]: {test_results}")
            continue

        full_image_model: tf.keras.Model = build_autoencoder(
            input_shape=FULL_IMAGE_INPUT_SHAPE
        )
        full_image_model.set_weights(model.get_weights())

        full_model_dir: Path = get_model_output_dir("denoising_autoencoder_full")
        full_model_save_path: Path = full_model_dir / f"{experiment_name}.keras"
        full_image_model.save(full_model_save_path)
        print(
            f"Saved full-image model [{name}_{experiment_name}] to: {full_model_save_path}"
        )

        full_image_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="mse",  # mean squared error
            metrics=["mae"],  # mean absolute error
        )

        avg_mse, avg_mae = evaluate_full_image_dataset(full_image_model, test_full_ds)
        print(f"Test results [{name}_{experiment_name}]: [{avg_mse}, {avg_mae}]")


if __name__ == "__main__":
    model_process()
