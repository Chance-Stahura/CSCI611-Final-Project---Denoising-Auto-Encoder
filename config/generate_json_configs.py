import json
from pathlib import Path

OUTPUT_DIR: Path = Path(__file__).resolve().parent

EPOCHS: list[int] = [5]
DATASETS: list[str] = ["cbsd68"]

SIGMA_VALUES: list[int] = [4, 10, 15, 20, 25]
P_VALUES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
SIZE_VALUES: list[int] = [4, 10, 15, 20, 25]


def write_config(config: dict, output_dir: Path) -> None:
    """Writes one config to a JSON file."""
    experiment_name: str = config["experiment"]["name"]
    filepath: Path = output_dir / f"{experiment_name}.json"

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def build_gaussian_config(sigma: int, epochs: int, dataset: str) -> dict:
    """Builds one gaussian config."""
    experiment_name: str = f"gaussian_{sigma}_{epochs}_{dataset}"

    return {
        "experiment": {
            "name": experiment_name,
        },
        "noise": {
            "type": "gaussian",
            "sigma": sigma,
        },
        "training": {
            "epochs": epochs,
            "dataset": dataset,
        },
    }


def build_salt_pepper_config(p: float, epochs: int, dataset: str) -> dict:
    """Builds one salt-and-pepper config."""
    p_label: str = str(p).replace(".", "p")
    experiment_name: str = f"salt_pepper_{p_label}_{epochs}_{dataset}"

    return {
        "experiment": {
            "name": experiment_name,
        },
        "noise": {
            "type": "salt_pepper",
            "p": p,
        },
        "training": {
            "epochs": epochs,
            "dataset": dataset,
        },
    }


def build_occlusion_config(size: int, epochs: int, dataset: str) -> dict:
    """Builds one occlusion config."""
    experiment_name: str = f"occlusion_{size}_{epochs}_{dataset}"

    return {
        "experiment": {
            "name": experiment_name,
        },
        "noise": {
            "type": "occlusion",
            "size": size,
        },
        "training": {
            "epochs": epochs,
            "dataset": dataset,
        },
    }


def generate_configs(output_dir: Path) -> None:
    """Generates all configs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for epochs in EPOCHS:
        for dataset in DATASETS:
            for sigma in SIGMA_VALUES:
                config: dict = build_gaussian_config(sigma, epochs, dataset)
                write_config(config, output_dir)

            for p in P_VALUES:
                config = build_salt_pepper_config(p, epochs, dataset)
                write_config(config, output_dir)

            for size in SIZE_VALUES:
                config = build_occlusion_config(size, epochs, dataset)
                write_config(config, output_dir)


if __name__ == "__main__":
    generate_configs(OUTPUT_DIR)
