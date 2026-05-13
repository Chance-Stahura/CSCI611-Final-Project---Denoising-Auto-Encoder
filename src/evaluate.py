import tensorflow as tf
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

from datetime import datetime, timedelta
import time

from auto_encoder import (
    build_image_set,
    TEST_DS,
    PATCH_SIZE,
    TEST_BATCH_SIZE,
    MODELS_DIR,
)

BASE_DIR: Path = Path(__file__).resolve().parents[1]

RESULTS_DIR: Path = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR: Path = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

from dataset import Dataset

matplotlib.use("Agg")

test_imgs = build_image_set(TEST_DS)


def compute_psnr(mse):
    return 10 * tf.math.log(1.0 / mse) / tf.math.log(10.0)


# depricated function for more efficient version
# def reconstruct_full_image(
#     model, noisy_img: np.ndarray, patch_size: int = 64
# ) -> np.ndarray:
#     height, width, channels = noisy_img.shape

#     stride = patch_size // 2

#    # pad_h = (patch_size - height % patch_size) % patch_size
#    # pad_w = (patch_size - width % patch_size) % patch_size
#     pad_h = (stride - height % stride) % stride
#     pad_w = (stride - width % stride) % stride

#     padded = np.pad(noisy_img, ((0, pad_h + patch_size), (0, pad_w + patch_size), (0, 0)), mode="reflect")
#     pad_height, pad_width, _ = padded.shape
#     output = np.zeros_like(padded)
#     weights = np.zeros_like(padded)

#     w = np.hanning(patch_size)
#     window = np.outer(w, w)
#     window = window[..., np.newaxis].astype(np.float32)

#     for i in range(0, pad_height - patch_size + 1, stride):
#         for j in range(0, pad_width - patch_size + 1, stride):
#             patch: np.ndarray = padded[
#                 i : i + patch_size,
#                 j : j + patch_size,
#                 :,
#             ][np.newaxis, ...]
#             pred = model(patch, training = False).numpy()[0]
#             output[i:i + patch_size, j:j + patch_size, :] += pred * window
#             weights[i:i + patch_size, j:j + patch_size, :] += window

#     output = output / np.maximum(weights, 1e-8)

#     return output[:height, :width, :]


def reconstruct_full_image(
    model,
    noisy_img: np.ndarray,
    patch_size: int = 64,
    batch_size: int = 128,
) -> np.ndarray:

    height, width, channels = noisy_img.shape

    stride = patch_size // 3

    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride

    padded = np.pad(
        noisy_img,
        ((0, pad_h + patch_size),
         (0, pad_w + patch_size),
         (0, 0)),
        mode="reflect"
    )

    pad_height, pad_width, _ = padded.shape

    output = np.zeros_like(padded, dtype=np.float32)
    weights = np.zeros_like(padded, dtype=np.float32)

    # Hann window
    w = np.hanning(patch_size)
    window = np.outer(w, w).astype(np.float32)
    window = window[..., np.newaxis]

    # ---------------------------------------------------
    # Extract all patches first
    # ---------------------------------------------------

    patches = []
    coords = []

    for i in range(0, pad_height - patch_size + 1, stride):
        for j in range(0, pad_width - patch_size + 1, stride):

            patch = padded[
                i:i + patch_size,
                j:j + patch_size,
                :
            ]

            patches.append(patch)
            coords.append((i, j))

    patches = np.array(patches, dtype=np.float32)

    # ---------------------------------------------------
    # Batched inference
    # ---------------------------------------------------

    preds = []

    for k in range(0, len(patches), batch_size):

        batch = patches[k:k + batch_size]

        pred_batch = model(batch, training=False).numpy()

        preds.append(pred_batch)

    preds = np.concatenate(preds, axis=0)

    # ---------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------

    for pred, (i, j) in zip(preds, coords):

        weighted = pred * window

        output[
            i:i + patch_size,
            j:j + patch_size,
            :
        ] += weighted

        weights[
            i:i + patch_size,
            j:j + patch_size,
            :
        ] += window

    output /= np.maximum(weights, 1e-8)

    return output[:height, :width, :]


def evaluate(
    experiment: str,
    noise_type: str,
    sigma: int,
    salt_pepper_p: float,
    occlusion_size: int,
    epochs: int,
) -> None:
    """Evaluates all saved models for one."""
    EXPERIMENT_DIR: Path = RESULTS_DIR / experiment
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    test_full_ds = Dataset(
        image_paths=test_imgs,
        patch_size=PATCH_SIZE,
        sigma=sigma,
        batch_size=TEST_BATCH_SIZE,
        training=False,
        return_full_image=True,
        shuffle=False,
        noise_type=noise_type,
        pad_multiple=4,
        salt_pepper_p=salt_pepper_p,
        occlusion_size=occlusion_size,
    )

    our_model = tf.keras.models.load_model(MODELS_DIR / f"denoise/{experiment}.keras")
    mlp_model = tf.keras.models.load_model(MODELS_DIR / f"dense/{experiment}.keras")
    tf_model = tf.keras.models.load_model(MODELS_DIR / f"benchmark/{experiment}.keras")

    models: dict[str, tf.keras.Model] = {
        "denoising_autoencoder": our_model,
        "dense_autoencoder": mlp_model,
        "original_benchmark": tf_model,
    }

    psnr_scores: dict[str, float] = {}
    ssim_scores: dict[str, float] = {}

    # use first image for comparison grid
    num_images = len(test_full_ds)
    noisy_batch, clean_batch = test_full_ds[25]  # choose index for plot outputs
    noisy_img = noisy_batch[0].numpy()
    clean_img = clean_batch[0].numpy()

    # run evaluation/testing on CBSD68
    for name, model in models.items():

        # time monitoring
        print(f"[{datetime.now().strftime('%H:%M:%S')}]  model: {name}")
        start_time = time.time()  # initial elapsed time = 00:00:00

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

        for i in range(num_images):
            noisy_batch_it, clean_batch_it = test_full_ds[i]
            noisy_img_it = noisy_batch_it[0].numpy()
            clean_img_it = clean_batch_it[0].numpy()

            # dense model + benchmark are not meant to handle full images
            if name == "dense_autoencoder" or name == "original_benchmark":
                pred_img_it = reconstruct_full_image(
                    model, noisy_img_it, patch_size=PATCH_SIZE
                )
            else:
                pred_img_it = model(noisy_batch_it, training=False).numpy()[0]

            # Crop prediction to clean image size
            pred_img_it = pred_img_it[
                :clean_img_it.shape[0],
                :clean_img_it.shape[1],
                :
            ]

            total_mse += float(np.mean((clean_img_it - pred_img_it) ** 2))
            pred_batch_it: np.ndarray = pred_img_it[np.newaxis, ...].astype(np.float32)
            total_ssim += float(
                tf.reduce_mean(
                    tf.image.ssim(clean_batch_it, pred_batch_it, max_val=1.0)
                ).numpy()
            )

        # time monitoring
        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(f"+[{elapsed}]  MSE, SSIM calculated")
        start_time = time.time()  # initial elapsed time = 00:00:00

        avg_mse: float = total_mse / num_images
        avg_psnr: float = float(compute_psnr(avg_mse).numpy())
        avg_ssim: float = total_ssim / num_images

        psnr_scores[name] = avg_psnr
        ssim_scores[name] = avg_ssim

        model_metrics = {
            "model": name_out,
            "noise_type": noise_type,
            # "noise_strength": noise_strength,
            "epochs": epochs,
            "mse": avg_mse,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
        }

        if noise_type == "gaussian":
            model_metrics["noise_strength"] = sigma
        elif noise_type == "salt_pepper":
            model_metrics["noise_strength"] = salt_pepper_p
        elif noise_type == "occlusion":
            model_metrics["noise_strength"] = occlusion_size
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # save metrics for each model to model_metrics.json
        out_path = EXPERIMENT_DIR / f"{name_out}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(model_metrics, f, indent=4)

        if name in {"dense_autoencoder", "original_benchmark"}:
            pred_img = reconstruct_full_image(
                model, noisy_img, patch_size=PATCH_SIZE
            )
        else:
            pred_img = model(noisy_batch, training=False).numpy()[0]

        # time monitoring
        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(f"+[{elapsed}]  Predicted image constructed")
        start_time = time.time()  # initial elapsed time = 00:00:00

        # generate comparison grids:
        # noisy input -> model output -> ground truth
        figure_size: tuple[int, int] = (15, 5)
        dpi_val: int = 300
        pad_pixels: int = 20
        padding_val: float = pad_pixels / dpi_val

        fig, axes = plt.subplots(1, 3, figsize=figure_size, dpi=dpi_val)

        title_fs = 28
        label_fs = 24

        suptitle: str = (
            f"Denoising Comparison: {name}\n"
            f"experiment: {experiment}"
        )
        plt.suptitle(suptitle, fontsize=title_fs)

        axes[0].imshow(noisy_img, interpolation="nearest")
        axes[0].set_title("Noisy", fontsize=label_fs)

        axes[1].imshow(pred_img, interpolation="nearest")
        axes[1].set_title("Denoised", fontsize=label_fs)

        axes[2].imshow(clean_img, interpolation="nearest")
        axes[2].set_title("Clean", fontsize=label_fs)

        for ax in axes:
            ax.axis("off")

        plt.imsave(EXPERIMENT_DIR / f"{name}_denoised.png", pred_img)
        if not (EXPERIMENT_DIR / "noisy.png").is_file():
            plt.imsave(EXPERIMENT_DIR / "noisy.png", noisy_img)
        if not (EXPERIMENT_DIR / "clean.png").is_file():
            plt.imsave(EXPERIMENT_DIR / "clean.png", clean_img)

        plt.tight_layout()
        plt.savefig(EXPERIMENT_DIR / f"{name}_comparison.png",
                    bbox_inches="tight", pad_inches=padding_val)
        plt.close()

        # plot training loss curves per model
        with open(
            MODELS_DIR / f"{name_out}/histories/{experiment}_history.json",
            mode="r",
            encoding="utf-8",
        ) as f:
            history: dict[str, list[float]] = json.load(f)

        plt.plot(history["loss"], label=f"{name} Train")
        plt.plot(history["val_loss"], label=f"{name} Validation")
        plt.title(f"{name} Training Loss\nexperiment: {experiment}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(EXPERIMENT_DIR / f"{name_out}_loss.png")
        plt.close()

        # time monitoring
        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(f"+[{elapsed}]  Model-specific plots saved")
        # print(f" [{datetime.now().strftime('%H:%M:%S')}]\n")

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
