import tensorflow as tf
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from auto_encoder import build_image_set, evaluate_full_image_dataset, cbsd_ground_truth, PATCH_SIZE, NOISE_SIGMA, TEST_BATCH_SIZE, BASE_DIR
from dataset import Dataset
matplotlib.use('Agg')

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

# load each model
our_model = tf.keras.models.load_model("models/denoising_autoencoder.keras")
mlp_model = tf.keras.models.load_model("models/dense_autoencoder.keras")
tf_model = tf.keras.models.load_model("models/original_benchmark.keras")

models = {"denoising_autoencoder": our_model, "dense_autoencoder": mlp_model, "original_benchmark": tf_model}


def compute_psnr(mse):
    return 10 * tf.math.log(1.0 / mse) / tf.math.log(10.0)


psnr_scores = {}
ssim_scores = {}

outputs_path: Path = BASE_DIR / "outputs"
outputs_path.mkdir(parents=True, exist_ok=True)

# run evaluation/testing on CBSD68
for name, model in models.items():

    # dense model + benchmark are not meant to handle full images
    if name == "dense_autoencoder" or name == "original_benchmark":
        avg_mse, avg_mae = evaluate_full_image_dataset(model, test_patch_ds)
        noisy_batch, clean_batch = test_patch_ds[0]
    else:
        # compare and log PSNR and SSIM
        avg_mse, avg_mae = evaluate_full_image_dataset(model, test_full_ds)
        noisy_batch, clean_batch = test_full_ds[0]
    psnr = compute_psnr(avg_mse)
    psnr_scores[name] = psnr.numpy()
    predictions = model(noisy_batch)
    ssim = tf.image.ssim(clean_batch, predictions, max_val=1.0)
    ssim_scores[name] = tf.reduce_mean(ssim).numpy()

    # generate comparison grids:
    # noisy input -> model output -> ground truth
    fix, axes = plt.subplots(1, 3)
    plt.suptitle(f"{name} Denoising Comparison")
    axes[0].imshow(noisy_batch[0])
    axes[0].set_title("Noisy")
    axes[0].axis('off')
    axes[1].imshow(predictions[0])
    axes[1].set_title("Denoised")
    axes[1].axis('off')
    axes[2].imshow(clean_batch[0])
    axes[2].set_title("Clean")
    axes[2].axis('off')
    plt.savefig(f"outputs/{name}_comparison.png")
    plt.figure()

    # plot training loss curves per model
    with open(f"histories/{name}_history.json", "r") as f:
        history = json.load(f)
    plt.plot(history['loss'], label=f'{name} Train')
    plt.plot(history['val_loss'], label=f'{name} Validation')
    plt.title(f"{name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"outputs/{name}_loss.png")
    plt.figure()

# save bar chart comparing PSNR/SSIM for all models
plt.bar(psnr_scores.keys(), psnr_scores.values())
plt.title("Peak Signal-to-Noise Ratio (PSNR)")
plt.ylabel("dB")
plt.savefig("outputs/psnr_comparison.png")
plt.figure()
plt.bar(ssim_scores.keys(), ssim_scores.values())
plt.title("Structural Similarity Index Measure (SSIM)")
plt.ylabel("Score (0-1)")
plt.savefig("outputs/ssim_comparison.png")
