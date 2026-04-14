"""This will hold the dataset class"""

# coding: utf-8

from numpy import ndarray
from PIL.Image import Image as PILImage

import tensorflow as tf

from keras.utils import load_img


class Dataset(tf.keras.utils.Sequence):
    """Dataset Class"""

    def __init__(
        self,
        image_paths: list[str],
        patch_size: int = 64,
        sigma: int = 25,
        batch_size: int = 32,
        training: bool = True,
    ) -> None:
        """Dataset Constructor"""
        # self.img_dims = []
        self.image_paths = image_paths
        # for i in self.image_paths:
        #     img: PILImage = load_img(i)
        #     self.img_dims.append((img.size))
        self.patch_size: int = patch_size
        self.sigma: float = sigma / 255.0
        self.batch_size: int = batch_size
        self.training: bool = training

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image_as_tensor(self, path: str) -> tf.Tensor:
        """Loads an image as a tensor"""
        img_tensor: PILImage = load_img(path)
        # convert to tensor: (N, H, W, C)
        tensor_array: ndarray = tf.keras.utils.img_to_array(img_tensor)
        tensor_array = tensor_array / 255.0
        return tf.expand_dims(tensor_array, axis=0)

    def _extract_patches(self, img_tensor: tf.Tensor) -> tf.Tensor:
        """Extracts patches from an image"""
        return tf.image.extract_patches(
            images=img_tensor,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

    def _add_noise(self, clean_patch: tf.Tensor) -> tf.Tensor:
        """Adds noise to a clean patch"""
        added_noise: tf.Tensor = tf.random.normal(
            shape=tf.shape(clean_patch), stddev=self.sigma
        )
        noisy_patch: tf.Tensor = tf.add(clean_patch, added_noise)
        return tf.clip_by_value(noisy_patch, 0, 1)

    def __getitem__(self, idx: int) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns a batch of noisy and clean patches."""
        # load image
        path: str = self.image_paths[idx]
        img_tensor: tf.Tensor = self._load_image_as_tensor(path)
        clean_patch: tf.Tensor = self._extract_patches(img_tensor)
        noisy_patch: tf.Tensor = self._add_noise(clean_patch)
        return (noisy_patch, clean_patch)


if __name__ == "__main__":
    print("You are running the dataset.py file! Wrong one!")
