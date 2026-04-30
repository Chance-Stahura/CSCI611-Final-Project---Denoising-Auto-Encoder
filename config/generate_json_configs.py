import json
from pathlib import Path
from itertools import product

# NOTE: adjust lists according to desired output
config = {
    "experiment": {"name": "<noise_type>_<sigma_val>_<epoch_val>_<dataset>"},
    "noise": {
        "type": ["gaussian", "salt_pepper", "occlusion"],
        "sigma": [5, 10, 15, 20, 25],
    },
    "training": {
        "epochs": 5,  # [5, 10, 20],
        "dataset": "cbsd68",  # ["cbsd68", "bsds500", "waterloo"]
    },
}


def extract_sweep_params(cfg):
    """
    Flatten nested config into sweepable parameters.
    Returns dict: {("model", "architecture"): [...], ...}
    """
    sweep = {}

    for section, params in cfg.items():
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, list):
                    sweep[(section, key)] = value

    return sweep


def build_config(base_cfg, combo_dict):
    """
    Reconstruct full nested config from one combination.
    """
    new_cfg = {}

    for section, params in base_cfg.items():
        new_cfg[section] = {}

        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, list):
                    new_cfg[section][key] = combo_dict[(section, key)]
                else:
                    new_cfg[section][key] = value

    return new_cfg


def generate_experiment_name(cfg):
    return (
        # f"{cfg['model']['architecture']}_"
        f"{cfg['noise']['type']}_"
        f"{cfg['noise']['sigma']}_"
        f"{cfg['training']['epochs']}_"
        f"{cfg['training']['dataset']}"
    )


def generate_configs(config, output_dir="configs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = extract_sweep_params(config)

    keys = list(sweep.keys())
    values = list(sweep.values())

    for i, combo in enumerate(product(*values)):
        combo_dict = dict(zip(keys, combo))

        cfg = build_config(config, combo_dict)

        # generate experiment name
        cfg["experiment"]["name"] = generate_experiment_name(cfg)

        filepath = output_dir / f"{cfg['experiment']['name']}.json"

        with filepath.open("w") as f:
            json.dump(cfg, f, indent=4)


OUTPUT_DIR: Path = Path(__file__).resolve().parent

# run
generate_configs(config, output_dir=OUTPUT_DIR)
