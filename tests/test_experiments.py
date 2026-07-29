import importlib
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ExperimentSmokeTest(unittest.TestCase):
    def test_yaml_configs_load_and_modules_import(self):
        config_paths = sorted((ROOT / "experiments").rglob("*.yaml"))
        config_paths.extend(sorted((ROOT / "results").rglob("run_config.yaml")))
        self.assertTrue(config_paths)

        for path in config_paths:
            with self.subTest(config=str(path)):
                with path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)

                self.assertIsInstance(config, dict)
                if "agent" in config:
                    importlib.import_module(config["agent"]["module"])
                if "env" in config and "module" in config["env"]:
                    importlib.import_module(config["env"]["module"])


if __name__ == "__main__":
    unittest.main()
