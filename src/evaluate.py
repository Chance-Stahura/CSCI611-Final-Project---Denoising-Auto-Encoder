import tensorflow as tf
from json import load
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from auto_encoder import (
    build_image_set,
    cbsd_ground_truth,
    PATCH_SIZE,
    NOISE_SIGMA,
    TEST_BATCH_SIZE,
    BASE_DIR,
)

from dataset import Dataset

matplotlib.use("Agg")

METRICS_DIR: Path = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

test_imgs = build_image_set(cbsd_ground_truth)


def compute_psnr(mse):
    return 10 * tf.math.log(1.0 / mse) / tf.math.log(10.0)


def reconstruct_full_image(
    model, noisy_img: np.ndarray, patch_size: int = 64
) -> np.ndarray:
    height, width, channels = noisy_img.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size

    padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pad_height, pad_width, _ = padded.shape
    output = np.zeros_like(padded)

    for i in range(0, pad_height, patch_size):
        for j in range(0, pad_width, patch_size):
            patch: np.ndarray = padded[
                i : i + patch_size,
                j : j + patch_size,
                :,
            ][np.newaxis, ...]
            output[i : i + patch_size, j : j + patch_size, :] = model(
                patch, training=False
            ).numpy()[0]

    return output[:height, :width, :]


def evaluate(
    experiment: str,
    noise_type: str,
    sigma: int,
    salt_pepper_p: float,
    occlusion_size: int,
) -> None:
    """Evaluates all saved models for one ."""
    EXPERIMENT_DIR: Path = METRICS_DIR / experiment
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    test_full_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        batch_size=1,
        training=False,
        return_full_image=True,
        shuffle=False,
        noise_type=noise_type,
        pad_multiple=4,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    test_patch_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=False,
        shuffle=False,
        noise_type=noise_type,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )
    our_model = tf.keras.models.load_model(f"outputs/denoise/{experiment}.keras")
    mlp_model = tf.keras.models.load_model(f"outputs/dense/{experiment}.keras")
    tf_model = tf.keras.models.load_model(f"outputs/benchmark/{experiment}.keras")

    models: dict[str, tf.keras.Model] = {
        "denoising_autoencoder": our_model,
        "dense_autoencoder": mlp_model,
        "original_benchmark": tf_model,
    }

    psnr_scores: dict[str, float] = {}
    ssim_scores: dict[str, float] = {}

    # run evaluation/testing on CBSD68
    for name, model in models.items():

        if name == "denoising_autoencoder":
            name_out = "denoise"
        elif name == "dense_autoencoder":
            name_out = "dense"
        elif name == "original_benchmark":
            name_out = "benchmark"
        else:
            raise ValueError(f"Unknown model name: {name}")

        # average PSNR and SSIM for all test images
        total_mse: float = 0.0
        total_ssim: float = 0.0
        num_images = len(test_full_ds)

        for i in range(num_images):
            noisy_batch, clean_batch = test_full_ds[i]
            noisy_img = noisy_batch[0].numpy()
            clean_img = clean_batch[0].numpy()

            # dense model + benchmark are not meant to handle full images
            if name == "dense_autoencoder" or name == "original_benchmark":
                pred_img = reconstruct_full_image(
                    model, noisy_img, patch_size=PATCH_SIZE
                )
            else:
                pred_img = model(noisy_batch, training=False).numpy()[0]

            total_mse += float(np.mean((clean_img - pred_img) ** 2))
            pred_batch: np.ndarray = pred_img[np.newaxis, ...].astype(np.float32)
            total_ssim += float(
                tf.reduce_mean(
                    tf.image.ssim(clean_batch, pred_batch, max_val=1.0)
                ).numpy()
            )

        avg_mse: float = total_mse / num_images
        avg_psnr: float = float(compute_psnr(avg_mse).numpy())
        avg_ssim: float = total_ssim / num_images

        psnr_scores[name] = avg_psnr
        ssim_scores[name] = avg_ssim

        # use first image for comparison grid
        noisy_batch, clean_batch = test_full_ds[0]
        noisy_img = noisy_batch[0].numpy()

        if name in {"dense_autoencoder", "original_benchmark"}:
            pred_img = reconstruct_full_image(
                model, noisy_img, patch_size=PATCH_SIZE
            )
        else:
            pred_img = model(noisy_batch, training=False).numpy()[0]

        # generate comparison grids:
        # noisy input -> model output -> ground truth
        fig, axes = plt.subplots(1, 3)
        plt.suptitle(f"{name} Denoising Comparison\nexperiment: {experiment}")

        # plt.title(f"Experiment: {experiment}")
        axes[0].imshow(noisy_batch[0])
        axes[0].set_title("Noisy")
        axes[0].axis("off")

        axes[1].imshow(pred_img)
        axes[1].set_title("Denoised")
        axes[1].axis("off")

        axes[2].imshow(clean_batch[0])
        axes[2].set_title("Clean")
        axes[2].axis("off")
        plt.savefig(EXPERIMENT_DIR / f"{name}_comparison.png")
        plt.close()

        # plot training loss curves per model
        with open(
            f"outputs/{name_out}/histories/{experiment}_history.json",
            mode="r",
            encoding="utf-8",
        ) as f:
            history: dict[str, list[float]] = load(f)

        plt.plot(history["loss"], label=f"{name} Train")
        plt.plot(history["val_loss"], label=f"{name} Validation")
        plt.title(f"{name} Training Loss\nexperiment: {experiment}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(EXPERIMENT_DIR / f"{name_out}_loss.png")
        plt.close()

    # save bar chart comparing PSNR/SSIM for all models
    plt.bar(psnr_scores.keys(), psnr_scores.values())
    plt.title(f"Peak Signal-to-Noise Ratio (PSNR)\nexperiment: {experiment}")
    plt.ylabel("dB")
    plt.savefig(EXPERIMENT_DIR / "psnr_comparison.png")
    plt.close()
    plt.bar(ssim_scores.keys(), ssim_scores.values())
    plt.title(
        f"Structural Similarity Index Measure (SSIM)\nexperiment: {experiment}"
    )
    plt.ylabel("Score (0-1)")
    plt.savefig(EXPERIMENT_DIR / "ssim_comparison.png")
    plt.close()

    return None
