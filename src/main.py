"""This file will run the auto_encoder code multiple times based from the config file."""

# coding: utf-8


from pathlib import Path
from shutil import move

from auto_encoder import model_process

BASE_DIR: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)
DONE_DIR: Path = BASE_DIR / "config/done"
DONE_DIR.mkdir(exist_ok=True)


def main() -> None:
    """The main code."""
    for path in CONFIG_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".json":
            print("Currently runnning with: {}", path)
            model_process()

            print("Done with: {}", path)

            try:
                if DONE_DIR.exists():
                    move(path, DONE_DIR)
            except PermissionError:
                print("Permission Error! Quitting...")
                raise


if __name__ == "__main__":
    pass
