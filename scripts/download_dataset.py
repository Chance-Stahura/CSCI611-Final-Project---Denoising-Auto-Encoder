import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress all logs

import zipfile
from pathlib import Path
import tensorflow as tf
import shutil
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("PROJECT_ROOT: ", PROJECT_ROOT)  # DEBUGGING
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TARGET_DIR_CBSD68 = DATA_DIR / "CBSD68"
TARGET_DIR_BSDS500 = DATA_DIR / "BSDS500"


def download_cbsd68():
    """Download CBSD68 dataset to data/."""

    # download + extract CBSD68 dataset
    cbsd68_url: str = (
        "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
    )
    # cbsd68_url = "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
    cbsd68_zip_path = Path(tf.keras.utils.get_file(
        fname="cbsd68.zip",
        origin=cbsd68_url,
        extract=False,
        cache_dir=str(PROJECT_ROOT),  # current directory
        cache_subdir="data"  # save to ./data/
    ))
    
    print(cbsd68_zip_path)
    
    extract_dir = cbsd68_zip_path.parent
    extracted_root = extract_dir / "CBSD68-dataset-master"

    if not extracted_root.exists():
        with zipfile.ZipFile(cbsd68_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    _safe_move(extracted_root / "CBSD68", extract_dir)
    _safe_rmtree(extracted_root)
    _safe_unlink(cbsd68_zip_path)  # deletes zip file

    # final image directory
    cbsd68_img_folder = TARGET_DIR_CBSD68 / "original_png"

    print("CBSD68 Dataset path:", cbsd68_img_folder)

    return cbsd68_img_folder


# def download_bsds500():


def _safe_move(path_target, path_destination):
    try:
        if path_target.exists() and path_destination.exists():
            shutil.move(path_target, path_destination)
    except PermissionError:
        pass


def _safe_rmtree(path, retries=5, delay=0.5):
    # print()
    # print(path)
    # print(path.exists())
    # print(path.is_file())
    for i in range(retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(delay)
    raise


def _safe_unlink(path):
    try:
        if path.exists():
            # print(path)
            # print(path.exists())
            # print(path.is_file())
            path.unlink()
    except PermissionError:
        pass


def get_cbsd68_path():
    if not TARGET_DIR_CBSD68.exists():
        return download_cbsd68()
    return TARGET_DIR_CBSD68


# def get_bsds500_path():
#     if not TARGET_DIR_BSDS500.exists():
#         return download_bsds500()
#     return TARGET_DIR_BSDS500


if __name__ == "__main__":
    cbsd68_img_folder = get_cbsd68_path()
    # bsds500_img_folder = get_bsds500_path()
