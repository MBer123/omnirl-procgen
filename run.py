import os

# Limit CPU thread usage before training
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
}

os.environ.update(THREAD_ENV)

import argparse
import importlib
import yaml

from omnirl.runtime.runner import Runner


def load_yaml(path):
    """Load a YAML file and return its parsed configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_module(module_path):
    """Import and return a Python module from its module path."""
    return importlib.import_module(module_path)


def parse_args():
    """Parse command-line arguments for runner, agent, and mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    return parser.parse_args()


def main():
    """Load configs, build the runner, and execute the selected mode."""
    args = parse_args()

    runner_config = load_yaml(args.runner_config)
    agent_config = load_yaml(args.agent_config)

    agent = import_module(agent_config["agent"]["module"])
    environment = import_module(runner_config["env"]["module"])

    runner = Runner.from_config(agent, environment, runner_config, agent_config)

    if args.mode == "train":
        with open(os.path.join(runner.ckpt_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump({**agent_config, **runner_config}, f, sort_keys=False)
        runner.fit()
    else:
        runner.eval()


if __name__ == "__main__":
    main()
