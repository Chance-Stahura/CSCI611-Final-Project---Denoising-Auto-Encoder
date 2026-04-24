"""This will download the CBSD68 and BSDS500 datasets."""

# coding: utf-8

import zipfile
from pathlib import Path
import tensorflow as tf
import shutil
import time
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress all logs

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT: ", PROJECT_ROOT)  # DEBUGGING

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CBSD68_URL: str = (
    "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
)
TARGET_DIR_CBSD68: Path = DATA_DIR / "CBSD68"

BSDS500_URL: str = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
TARGET_DIR_BSDS500: Path = DATA_DIR / "BSDS500"


def download_dataset(dataset_path: Path) -> Path:
    """Downloads the dataset if it doesn't exist"""

    dataset_url: str = ""
    target_dir: Path = Path()
    root_name: str = ""
    src_path: Path = Path()

    if dataset_path == TARGET_DIR_CBSD68:
        dataset_url = CBSD68_URL
        root_name = "CBSD68-dataset-master"
        target_dir = TARGET_DIR_CBSD68
    elif dataset_path == TARGET_DIR_BSDS500:
        dataset_url = BSDS500_URL
        root_name = "BSDS500-master"
        target_dir = TARGET_DIR_BSDS500
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")

    if target_dir.exists():
        print(f"Target directory {target_dir} already exists")
        return target_dir

    print(f"Downloading dataset from {dataset_url}")

    dataset_zip_path: Path = Path(
        tf.keras.utils.get_file(
            fname="dataset.zip",
            origin=dataset_url,
            extract=False,
            cache_dir=str(PROJECT_ROOT),  # current directory
            cache_subdir="data",  # save to ./data/
        )
    )

    extract_dir: Path = dataset_zip_path.parent
    extracted_root: Path = Path(f"{extract_dir}/{root_name}")

    if not extracted_root.exists():
        with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    if dataset_path == TARGET_DIR_CBSD68:
        src_path = extracted_root / "CBSD68"
    elif dataset_path == TARGET_DIR_BSDS500:
        src_path = extracted_root / "BSDS500" / "data" / "images"
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")

    if not target_dir.exists():
        _safe_move(src_path, target_dir)

    _remove_other_files(target_dir)
    _safe_rmtree(extracted_root)
    _safe_unlink(dataset_zip_path)  # deletes whatever.zip

    print(f"{root_name} Dataset path: {target_dir}")
    return target_dir

def _remove_other_files(root: Path) -> None:
    """Recursively delete all files that are not .jpg or .png (image files)"""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".jpg" and path.suffix.lower() != ".png":
            path.unlink()

def _safe_move(path_src: Path, path_dst: Path) -> None:
    """Moves path_src to path_dst"""
    try:
        if path_src.exists():
            shutil.move(str(path_src), str(path_dst))
    except PermissionError:
        pass


def _safe_rmtree(path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """Removes path recursively"""
    for _ in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(delay)
    raise


def _safe_unlink(path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """Removes path recursively"""
    for _ in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except (PermissionError, OSError):
            time.sleep(delay)
    raise


def _remove_thumbs_db(root: Path) -> None:
    """Recursively remove all Thumbs.db files under root."""
    if not root.exists():
        return

    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == "thumbs.db":
            _safe_unlink(path)


def get_path(dataset_path: Path) -> Path:
    if not dataset_path.exists():
        return download_dataset(dataset_path)
    return dataset_path


if __name__ == "__main__":
    # cbsd68_img_folder: Path = get_cbsd68_path()
    # bsds500_img_folder: Path = get_bsds500_path()
    cbsd68_img_folder: Path = get_path(TARGET_DIR_CBSD68)
    bsds500_img_folder: Path = get_path(TARGET_DIR_BSDS500)
