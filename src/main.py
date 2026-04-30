"""This file will run the auto_encoder code multiple times based from the config file."""

# coding: utf-8

from json import load

from pathlib import Path
from shutil import move

from dataset import (
    DEFAULT_SIGMA,
    DEFAULT_SALT_PEPPER_P,
    DEFAULT_OCCLUSION_SIZE,
)
from auto_encoder import model_process
from evaluate import evaluate

BASE_DIR: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)
DONE_DIR: Path = BASE_DIR / "config/done"
DONE_DIR.mkdir(exist_ok=True)


def main() -> None:
    """The main code. This will iterate through the config folder and run the code for each config file."""

    try:
        if not DONE_DIR.exists():
            DONE_DIR.mkdir(exist_ok=True)
    except PermissionError:
        print("Permission Error! Quitting...")
        raise

    for path in CONFIG_DIR.glob("*.json"):
        if path.is_file() and path.suffix.lower() == ".json":

            # dict[str, dict[str, (str | int)]]]
            config = {}
            with path.open("r", encoding="utf-8") as json_file:
                config = load(json_file)

            experiment_name: str = config["experiment"]["name"]
            noise_type: str = config["noise"]["type"]

            sigma: int = DEFAULT_SIGMA
            salt_pepper_p: float = DEFAULT_SALT_PEPPER_P
            occlusion_size: int = DEFAULT_OCCLUSION_SIZE

            if noise_type == "gaussian":
                sigma: int = config["noise"]["sigma"]
            elif noise_type == "salt_pepper":
                salt_pepper_p: float = config["noise"]["p"]
            elif noise_type == "occlusion":
                occlusion_size: int = config["noise"]["size"]

            epochs: int = config["training"]["epochs"]
            dataset: str = config["training"]["dataset"]

            print(f"\n{'=' * 50}")
            print(f"\nCurrently runnning experiment: {path}\n")

            print("\n>>> Building models, creating histories...\n")
            model_process(
                experiment_name=experiment_name,
                noise_type=noise_type,
                sigma=sigma,
                epochs=epochs,
                dataset=dataset,
                salt_pepper_p=salt_pepper_p,
                occlusion_size=occlusion_size,
            )

            print("\n>>> Evaluating experiment models...\n")
            evaluate(
                experiment_name,
                noise_type=noise_type,
                sigma=sigma,
                salt_pepper_p=salt_pepper_p,
                occlusion_size=occlusion_size,
            )

            print(f"Done with experiment: {path}")

            try:
                move(path, DONE_DIR)
            except PermissionError:
                print("Permission Error! Quitting...")
                raise

            print("Moved into done folder.\n\n", path)


if __name__ == "__main__":
    main()
