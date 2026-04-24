"""This will hold the dataset class."""

# coding: utf-8

import math

import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img

from .noise import add_gaussian_noise, add_occlusion, add_salt_pepper_noise

DEFAULT_PATCH_SIZE: int = 64
DEFAULT_SIGMA: int = 25
DEFAULT_BATCH_SIZE: int = 32

PIXEL_SCALE: float = 255.0
PIXEL_MIN: float = 0.0
PIXEL_MAX: float = 1.0
IMAGE_CHANNELS: int = 3

DEFAULT_NOISE_TYPE: str = "gaussian"
DEFAULT_SALT_PEPPER_P: float = 0.1
DEFAULT_OCCLUSION_SIZE: int = 12
GAUSSIAN_MEAN: float = 0.0


class Dataset(tf.keras.utils.Sequence):
    """Dataset Class"""

    patch_size: int = 64
    batch_size: int = 32

    def __init__(
        self,
        image_paths: list[str],
        patch_size: int = DEFAULT_PATCH_SIZE,
        sigma: int = DEFAULT_SIGMA,
        batch_size: int = DEFAULT_BATCH_SIZE,
        training: bool = True,
        return_full_image: bool = False,
        shuffle: bool = True,
        noise_type: str = DEFAULT_NOISE_TYPE,
        salt_pepper_p: float = DEFAULT_SALT_PEPPER_P,
        occlusion_size: int = DEFAULT_OCCLUSION_SIZE,
    ) -> None:
        """Dataset Constructor"""
        self.image_paths: list[str] = sorted(image_paths)
        self.patch_size: int = patch_size
        self.sigma: float = sigma / PIXEL_SCALE
        self.batch_size: int = batch_size
        self.training: bool = training
        self.return_full_image: bool = return_full_image
        self.shuffle: bool = shuffle
        self.noise_type: str = noise_type
        self.salt_pepper_p: float = salt_pepper_p
        self.occlusion_size: int = occlusion_size

        self.indexes: np.ndarray = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Returns the number of batches per epoch"""
        return math.ceil(len(self.image_paths) / self.batch_size)

    def on_epoch_end(self) -> None:
        """Shuffles indexes after each epoch"""
        if self.training and self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_image_as_tensor(self, path: str) -> tf.Tensor:
        """Loads an image as a tensor with shape (H, W, C)"""
        img_tensor = load_img(path)
        tensor_array: np.ndarray = img_to_array(img_tensor) / PIXEL_SCALE
        return tf.convert_to_tensor(tensor_array, dtype=tf.float32)

    def _random_crop(self, img_tensor: tf.Tensor) -> tf.Tensor:
        """Extracts a random patch from an image"""
        height: tf.Tensor = tf.shape(img_tensor)[0]
        width: tf.Tensor = tf.shape(img_tensor)[1]

        if height < self.patch_size or width < self.patch_size:
            new_height: tf.Tensor = tf.maximum(height, self.patch_size)
            new_width: tf.Tensor = tf.maximum(width, self.patch_size)
            img_tensor = tf.image.resize(img_tensor, [new_height, new_width])

        return tf.image.random_crop(
            img_tensor,
            size=[self.patch_size, self.patch_size, IMAGE_CHANNELS],
        )

    def _center_crop(self, img_tensor: tf.Tensor) -> tf.Tensor:
        """Extracts a center patch from an image"""
        height: tf.Tensor = tf.shape(img_tensor)[0]
        width: tf.Tensor = tf.shape(img_tensor)[1]

        if height < self.patch_size or width < self.patch_size:
            new_height: tf.Tensor = tf.maximum(height, self.patch_size)
            new_width: tf.Tensor = tf.maximum(width, self.patch_size)
            img_tensor = tf.image.resize(img_tensor, [new_height, new_width])

        return tf.image.resize_with_crop_or_pad(
            img_tensor,
            target_height=self.patch_size,
            target_width=self.patch_size,
        )

    def _apply_noise(self, clean_tensor: tf.Tensor) -> tf.Tensor:
        """Selects which noise to apply"""
        if self.noise_type == "gaussian":
            return add_gaussian_noise(clean_tensor)

        if self.noise_type == "salt_pepper":
            return add_salt_pepper_noise(clean_tensor)

        if self.noise_type == "occlusion":
            return add_occlusion(clean_tensor)

        raise ValueError(f"Unknown noise type: {self.noise_type}")

    def __getitem__(self, img_id: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns a batch of noisy and clean images/patches"""
        start: int = img_id * self.batch_size
        end: int = (img_id + 1) * self.batch_size
        batch_indices: np.ndarray = self.indexes[start:end]
        batch_paths: list[str] = []

        for counter in batch_indices:
            batch_paths.append(self.image_paths[counter])

        clean_batch: list[tf.Tensor] = []
        noisy_batch: list[tf.Tensor] = []

        for path in batch_paths:
            img_tensor: tf.Tensor = self._load_image_as_tensor(path)

            clean_tensor: tf.Tensor

            if self.return_full_image:
                clean_tensor = img_tensor
            else:
                if self.training:
                    clean_tensor = self._random_crop(img_tensor)
                else:
                    clean_tensor = self._center_crop(img_tensor)

            noisy_tensor: tf.Tensor = self._apply_noise(clean_tensor)

            clean_batch.append(clean_tensor)
            noisy_batch.append(noisy_tensor)

        noisy_batch_tensor: tf.Tensor = tf.stack(noisy_batch)
        clean_batch_tensor: tf.Tensor = tf.stack(clean_batch)

        return noisy_batch_tensor, clean_batch_tensor


if __name__ == "__main__":
    print("You are running the dataset.py file! Wrong one!")
