"""This file will run the auto_encoder code multiple times based from the config file."""

# coding: utf-8

from json import load

from pathlib import Path
from shutil import move

from auto_encoder import model_process

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
            model_name: str = config["model"]["architecture"]
            noise_type: str = config["noise"]["type"]
            sigma: int = config["noise"]["sigma"]
            epochs: int = config["training"]["epochs"]
            dataset: str = config["training"]["dataset"]

            print(f"Currently runnning with: {path}")
            model_process(
                experiment_name=experiment_name,
                model_name=model_name,
                noise_type=noise_type,
                sigma=sigma,
                epochs=epochs,
                dataset=dataset,
            )

            print(f"Done with: {path}")

            try:
                move(path, DONE_DIR)
            except PermissionError:
                print("Permission Error! Quitting...")
                raise

            print("Moved into done folder.\n\n", path)


if __name__ == "__main__":
    main()
