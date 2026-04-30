import tensorflow as tf
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from auto_encoder import build_image_set, cbsd_ground_truth, PATCH_SIZE, NOISE_SIGMA, TEST_BATCH_SIZE, BASE_DIR
from dataset import Dataset
matplotlib.use('Agg')

METRICS_DIR: Path = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

test_imgs = build_image_set(cbsd_ground_truth)

test_full_ds = Dataset(
    image_paths=test_imgs,
    patch_size=PATCH_SIZE,
    sigma=NOISE_SIGMA,
    batch_size=1,
    training=False,
    return_full_image=True,
    shuffle=False,
    noise_type="gaussian",
    pad_multiple=4,
)

test_patch_ds = Dataset(
    image_paths=test_imgs,
    patch_size=PATCH_SIZE,
    sigma=NOISE_SIGMA,
    batch_size=TEST_BATCH_SIZE,
    training=False,
    return_full_image=False,
    shuffle=False,
)


def compute_psnr(mse):
    return 10 * tf.math.log(1.0 / mse) / tf.math.log(10.0)


def reconstruct_full_image(model, noisy_img: np.ndarray, patch_size: int = 64) -> np.ndarray:
    H, W, C = noisy_img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pH, pW, _ = padded.shape
    output = np.zeros_like(padded)

    for i in range(0, pH, patch_size):
        for j in range(0, pW, patch_size):
            patch = padded[i:i+patch_size, j:j+patch_size, :][np.newaxis, ...]
            output[i:i+patch_size, j:j+patch_size, :] = model(patch, training=False).numpy()[0]

    return output[:H, :W, :]


def evaluate(experiment = str) -> None:

    EXPERIMENT_DIR: Path = METRICS_DIR / experiment
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    our_model = tf.keras.models.load_model(f"outputs/denoise/{experiment}.keras")
    mlp_model = tf.keras.models.load_model(f"outputs/dense/{experiment}.keras")
    tf_model = tf.keras.models.load_model(f"outputs/benchmark/{experiment}.keras")

    models = {
        "denoising_autoencoder": our_model,
        "dense_autoencoder": mlp_model,
        "original_benchmark": tf_model
    }

    psnr_scores = {}
    ssim_scores = {}

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

        #average PSNR and SSIM for all test images
        total_mse = 0.0
        total_ssim = 0.0
        n = len(test_full_ds)

        for i in range(n):
            noisy_batch, clean_batch = test_full_ds[i]
            noisy_img = noisy_batch[0].numpy()
            clean_img = clean_batch[0].numpy()

            # dense model + benchmark are not meant to handle full images
            if name == "dense_autoencoder" or name == "original_benchmark":
                pred_img = reconstruct_full_image(model, noisy_img, patch_size=PATCH_SIZE)
            else:
                pred_img = model(noisy_batch, training=False).numpy()[0]

            total_mse += np.mean((clean_img - pred_img) ** 2)
            pred_batch = pred_img[np.newaxis, ...].astype(np.float32)
            total_ssim += tf.reduce_mean(tf.image.ssim(clean_batch, pred_batch, max_val = 1.0)).numpy()

        avg_mse = total_mse / n
        psnr = compute_psnr(avg_mse)
        psnr_scores[name] = psnr.numpy()
        ssim_scores[name] = total_ssim / n

        # use first image for comparison grid
        noisy_batch, clean_batch = test_full_ds[0]
        noisy_img = noisy_batch[0].numpy()

        if name == "dense_autoencoder" or name == "original_benchmark":
            pred_img = reconstruct_full_image(model, noisy_img, patch_size = PATCH_SIZE)
        else:
            pred_img = model(noisy_batch, training = False).numpy()[0]

        # generate comparison grids:
        # noisy input -> model output -> ground truth
        fig, axes = plt.subplots(1, 3)
        plt.suptitle(f"{name} Denoising Comparison")
        # plt.title(f"Experiment: {experiment}")
        axes[0].imshow(noisy_batch[0])
        axes[0].set_title("Noisy")
        axes[0].axis('off')
        axes[1].imshow(pred_img)
        axes[1].set_title("Denoised")
        axes[1].axis('off')
        axes[2].imshow(clean_batch[0])
        axes[2].set_title("Clean")
        axes[2].axis('off')
        plt.savefig(EXPERIMENT_DIR/f"{name}_comparison.png")
        plt.close()

        # plot training loss curves per model
        with open(f"outputs/{name_out}/histories/{experiment}_history.json", "r") as f:
            history = json.load(f)
        plt.plot(history['loss'], label=f'{name} Train')
        plt.plot(history['val_loss'], label=f'{name} Validation')
        plt.title(f"{name} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(EXPERIMENT_DIR/f"{name_out}_loss.png")
        plt.close()

    # save bar chart comparing PSNR/SSIM for all models
    plt.bar(psnr_scores.keys(), psnr_scores.values())
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.ylabel("dB")
    plt.savefig(EXPERIMENT_DIR/"psnr_comparison.png")
    plt.close()
    plt.bar(ssim_scores.keys(), ssim_scores.values())
    plt.title("Structural Similarity Index Measure (SSIM)")
    plt.ylabel("Score (0-1)")
    plt.savefig(EXPERIMENT_DIR/"ssim_comparison.png")
    plt.close()

    return None
