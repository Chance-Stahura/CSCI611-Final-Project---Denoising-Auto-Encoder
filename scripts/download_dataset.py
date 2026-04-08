from pathlib import Path
import tensorflow as tf
import shutil

def download_cbsd68(BASE_DIR: Path):
 
    # Data directory
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)

    # download + extract CBSD68 dataset
    cbsd68_url = 'https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip'
    cbsd68_zip_path = tf.keras.utils.get_file(
        fname='cbsd68.zip',
        origin=cbsd68_url,
        extract=True,
        cache_dir = '.', # current directory
        cache_subdir = 'data' # save to ./data/
    )

    cbsd68_zip_path = Path(cbsd68_zip_path)

    # restructure directories
    extracted_base = cbsd68_zip_path.parent / "cbsd68_extracted"
    extracted_root = extracted_base / "CBSD68-dataset-master"
    # extracted_root = cbsd68_zip_path.parent / "cbsd68_extracted" / "CBSD68-dataset-master"
    inner_dataset = extracted_root / "CBSD68"
    target_dataset = DATA_DIR / "CBSD68"

    if inner_dataset.exists() and not target_dataset.exists():
        shutil.move(str(inner_dataset), str(target_dataset))

    if extracted_root.exists():
        shutil.rmtree(extracted_root, ignore_errors=True)

    # final image directory
    cbsd68_img_folder = target_dataset / "original_png"

    print("CBSD68 Dataset path:", cbsd68_img_folder)

    # delete "cbsd68_extracted" and .zip
    try:
        # extracted_base.unlink()
        shutil.rmtree(extracted_base)
    except FileNotFoundError:
        pass
    try:
        Path(DATA_DIR / 'cbsd68.zip').unlink()
    except FileNotFoundError:
        print("File not found:", Path(DATA_DIR / 'cbsd68.zip'))
        pass
    
    return cbsd68_img_folder

# def download_bsd500(BASE_DIR: Path):
    # TODO
    # return bsd500_img_folder

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    download_cbsd68(BASE_DIR)
    # download_bsd500(BASE_DIR)
