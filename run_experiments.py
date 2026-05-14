import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(config: dict):
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(flatten_dict(value))
        else:
            flat[key] = value
    return flat


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments from a sweep file.")
    parser.add_argument("--sweep", type=str, required=True, help="path to the sweep yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--dry_run", action="store_true", help="only print generated commands")
    args = parser.parse_args()

    sweep_path = Path(args.sweep).resolve()
    sweep = load_yaml(sweep_path)

    base_config_path = (sweep_path.parent / sweep["base_config"]).resolve()
    base_config = load_yaml(base_config_path)
    experiments = sweep.get("experiments", [])
    generated_dir = Path(sweep.get("generated_dir", "configs/generated")).resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:
        name = experiment["name"]
        merged = deep_update(base_config, experiment.get("overrides", {}))
        merged["model_name"] = merged.get("model_name", name)
        merged.setdefault("ckpt", "checkpoints")
        merged.setdefault("output_dir", "videos")

        config_path = generated_dir / f"{name}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(flatten_dict(merged), f, sort_keys=False, allow_unicode=True)

        cmd = [sys.executable, f"{args.mode}.py", "--config_file", str(config_path)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
