"""This holds the fucntion of adding noise into images."""

# coding: utf-8

from typing import Callable

import tensorflow as tf


def add_gaussian_noise(x: tf.Tensor, sigma: float = 0.2) -> tf.Tensor:
    """Adds gaussian noise to an image"""
    noise: tf.Tensor = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=sigma)
    x_noisy: tf.Tensor = x + noise
    return tf.clip_by_value(x_noisy, 0.0, 1.0)


def add_salt_pepper_noise(x: tf.Tensor, p: float = 0.1) -> tf.Tensor:
    """Adds salt and pepper noise to an image"""
    random_vals: tf.Tensor = tf.random.uniform(tf.shape(x))

    salt: tf.Tensor = tf.cast(random_vals > (1 - p / 2), tf.float32)
    pepper: tf.Tensor = tf.cast(random_vals < (p / 2), tf.float32)

    x_noisy: tf.Tensor = x * (1 - salt - pepper) + salt
    return x_noisy


def add_occlusion(x: tf.Tensor, size: int = 12) -> tf.Tensor:
    """Adds random square occlusion to an image."""
    h: tf.Tensor = tf.shape(x)[0]
    w: tf.Tensor = tf.shape(x)[1]
    c: tf.Tensor = tf.shape(x)[2]

    occ_size: tf.Tensor = tf.minimum(tf.cast(size, tf.int32), tf.minimum(h, w))

    top: tf.Tensor = tf.random.uniform([], 0, h - occ_size + 1, dtype=tf.int32)
    left: tf.Tensor = tf.random.uniform([], 0, w - occ_size + 1, dtype=tf.int32)

    pad_bottom: tf.Tensor = h - top - occ_size
    pad_right: tf.Tensor = w - left - occ_size

    zero_block: tf.Tensor = tf.zeros([occ_size, occ_size, c], dtype=x.dtype)

    mask: tf.Tensor = tf.pad(
        zero_block,
        paddings=[
            [top, pad_bottom],
            [left, pad_right],
            [0, 0],
        ],
        mode="CONSTANT",
        constant_values=1,
    )

    return x * mask


def add_multi_occlusion(x: tf.Tensor, size: int = 12, k: int = 3) -> tf.Tensor:
    """Add k many occlusion sqaures."""
    for _ in range(k):
        x = add_occlusion(x, size)
    return x


def get_noise_fn(config: dict) -> Callable[[tf.Tensor], tf.Tensor]:
    """Returns a function that adds noise to an image"""
    noise_type = config["noise"]["type"]

    if noise_type == "gaussian":
        sigma = config["noise"]["sigma"]
        return lambda x: add_gaussian_noise(x, sigma)

    elif noise_type == "salt_pepper":
        p = config["noise"]["p"]
        return lambda x: add_salt_pepper_noise(x, p)

    elif noise_type == "occlusion":
        size = config["noise"]["size"]
        return lambda x: add_occlusion(x, size)

    else:
        raise ValueError("Unknown noise type")


if __name__ == "__main__":
    print("You are running the noise.py file! Wrong one!")
