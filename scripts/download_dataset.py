from pathlib import Path
import tensorflow as tf
import shutil
import time
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TARGET_DIR_CBSD68 = DATA_DIR / "CBSD68"
TARGET_DIR_BSDS500 = DATA_DIR / "BSDS500"


def download_cbsd68():
    """Download CBSD68 dataset to data/."""

    # Data directory

    # download + extract CBSD68 dataset
    cbsd68_url: str = (
        "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"
    )
    cbsd68_zip_path = tf.keras.utils.get_file(
        fname="cbsd68.zip",
        origin=cbsd68_url,
        extract=True,
        cache_dir=str(PROJECT_ROOT),  # current directory
        cache_subdir="data"  # save to ./data/
    )

    cbsd68_zip_path = Path(cbsd68_zip_path)
    print(cbsd68_zip_path)

    # restructure directories
    extracted_root = cbsd68_zip_path.parent / "cbsd68_extracted"
    extracted_branch = extracted_root / "CBSD68-dataset-master"
    # extracted_root = cbsd68_zip_path.parent / "cbsd68_extracted" / "CBSD68-dataset-master"
    inner_dataset = extracted_branch / "CBSD68"
    
    _safe_rmtree(extracted_root)
    # _safe_unlink(cbsd68_zip_path)  # meant to delete zip file; not working

    if inner_dataset.exists() and not TARGET_DIR_CBSD68.exists():
        shutil.move(str(inner_dataset), str(TARGET_DIR_CBSD68))

    if extracted_branch.exists():
        shutil.rmtree(extracted_branch, ignore_errors=True)

    # final image directory
    cbsd68_img_folder = TARGET_DIR_CBSD68 / "original_png"

    print("CBSD68 Dataset path:", cbsd68_img_folder)

    # delete "cbsd68_extracted" and .zip
    # try:
    #     # extracted_root.unlink()
    #     shutil.rmtree(extracted_root)
    # except FileNotFoundError:
    #     pass
    # try:
    #     Path(DATA_DIR / 'cbsd68.zip').unlink()
    # except FileNotFoundError:
    #     print("File not found:", Path(DATA_DIR / 'cbsd68.zip'))
    #     pass

    return cbsd68_img_folder


# def download_bsds500():
#     """Download BSDS500 dataset to data/."""

#     # Data directory

#     # download + extract BSDS500 dataset
#     bsds500_url: str = (
#         "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
#     )
#     bsds500_zip_path = tf.keras.utils.get_file(
#         fname="BSDS500.zip",
#         origin=bsds500_url,
#         extract=True,
#         cache_dir=str(PROJECT_ROOT),  # current directory
#         cache_subdir="data"  # save to ./data/
#     )

#     bsds500_zip_path = Path(bsds500_zip_path)

#     # restructure directories
#     # extracted_root = bsds500_zip_path.parent / "bsds500_extracted"
#     # extracted_branch = extracted_root / "BSDS500-dataset-master"
#     # inner_dataset = extracted_branch / "BSDS500"
    
#     # _safe_rmtree(extracted_root)
#     # # _safe_unlink(bsds500_zip_path)  # meant to delete zip file; not working

#     # if inner_dataset.exists() and not TARGET_DIR_CBSD68.exists():
#     #     shutil.move(str(inner_dataset), str(TARGET_DIR_CBSD68))

#     # if extracted_branch.exists():
#     #     shutil.rmtree(extracted_branch, ignore_errors=True)

#     # final image directory
#     # bsds500_img_folder = TARGET_DIR_CBSD68 / "original_png"

#     # print("BSDS500 Dataset path:", bsds500_img_folder)

#     return bsds500_zip_path  # bsds500_img_folder


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
    # cbsd68_img_folder = download_cbsd68()  # for debugging
    cbsd68_img_folder = get_cbsd68_path()
    # bsds500_img_folder = get_bsds500_path()
