import importlib
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from omnirl.networks import ImpalaCNN, SPPOCNN
from omnirl.runtime.runner import Runner


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
ALGORITHMS = ["ppo", "sppo"]
NETWORKS = {
    "omnirl.algorithms.ppo": ImpalaCNN,
    "omnirl.algorithms.sppo": SPPOCNN,
}


def load_yaml(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class StructureTest(unittest.TestCase):
    def test_algorithm_exports(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                module = importlib.import_module(f"omnirl.algorithms.{name}")
                for component in ["Learner", "Memory", "Policy"]:
                    self.assertTrue(hasattr(module, component))

                learner = module.Learner.__ray_metadata__.modified_class
                for method in [
                    "get_weights",
                    "get_version",
                    "learn",
                    "save_checkpoint",
                    "load_checkpoint",
                    "load_model",
                ]:
                    self.assertTrue(callable(getattr(learner, method)))

                memory = module.Memory.__ray_metadata__.modified_class
                for method in ["add", "get_batch", "on_learn", "ready", "stats"]:
                    self.assertTrue(callable(getattr(memory, method)))

                for method in ["set_weights", "choose_action", "sample_action"]:
                    self.assertTrue(callable(getattr(module.Policy, method)))

    def test_environment_backend_exports(self):
        environment = importlib.import_module("omnirl.envs.procgen")
        for component in [
            "CollectionLoop",
            "EvaluationLoop",
            "aggregate_evaluations",
            "build_env",
            "get_env_spec",
        ]:
            self.assertTrue(hasattr(environment, component))
        self.assertTrue(callable(environment.CollectionLoop.collect))
        self.assertTrue(callable(environment.EvaluationLoop.run))

    def test_config_module_paths(self):
        paths = list(EXPERIMENTS.rglob("*.yaml"))
        self.assertTrue(paths)

        for path in paths:
            with self.subTest(config=str(path)):
                config = load_yaml(path)
                self.assertNotIn("pipeline", config)
                if "agent" in config:
                    importlib.import_module(config["agent"]["module"])
                else:
                    importlib.import_module(config["env"]["module"])

    def test_config_fields_match_runtime_contracts(self):
        runner_params = set(inspect.signature(Runner.__init__).parameters)

        for path in EXPERIMENTS.rglob("*.yaml"):
            with self.subTest(config=str(path)):
                config = load_yaml(path)

                if "agent" in config:
                    self.assertEqual(set(config), {"agent", "modules"})
                    self.assertEqual(set(config["agent"]), {"module"})
                    self.assertLessEqual(
                        set(config["modules"]),
                        {"learner", "memory", "policy", "network"},
                    )

                    module = importlib.import_module(config["agent"]["module"])
                    components = {
                        "learner": module.Learner.__ray_metadata__.modified_class,
                        "memory": module.Memory.__ray_metadata__.modified_class,
                        "policy": module.Policy,
                    }
                    for name, component in components.items():
                        kwargs = set((config["modules"].get(name) or {}).keys())
                        params = set(inspect.signature(component.__init__).parameters)
                        self.assertLessEqual(kwargs, params)

                    network_kwargs = set(
                        (config["modules"].get("network") or {}).keys()
                    )
                    network_params = set(
                        inspect.signature(
                            NETWORKS[config["agent"]["module"]].__init__
                        ).parameters
                    )
                    self.assertLessEqual(network_kwargs, network_params)
                else:
                    self.assertLessEqual(
                        set(config),
                        {"env", "runner", "device", "seed"},
                    )
                    self.assertIn("module", config["env"])
                    self.assertNotIn("role", config["env"])
                    self.assertLessEqual(set(config["runner"]), runner_params)
                    self.assertLessEqual(
                        set(config.get("device", {})),
                        {"learner"},
                    )
                    self.assertLessEqual(
                        set(config.get("seed", {})),
                        {"learner", "sampler", "evaluator"},
                    )

    def test_runner_config_loading(self):
        train_path = next(iter(sorted(EXPERIMENTS.rglob("train.yaml"))))
        experiment_dir = train_path.parent
        agent_path = next(
            path
            for path in sorted(experiment_dir.glob("*.yaml"))
            if path.name not in {"train.yaml", "test.yaml"}
        )
        runner_config = load_yaml(train_path)
        agent_config = load_yaml(agent_path)

        agent = object()
        environment = object()
        with patch.object(Runner, "__init__", return_value=None) as init:
            Runner.from_config(agent, environment, runner_config, agent_config)
        captured_kwargs = init.call_args.kwargs

        expected_env_config = dict(runner_config["env"])
        expected_env_config.pop("module")
        self.assertIs(captured_kwargs["agent"], agent)
        self.assertIs(captured_kwargs["environment"], environment)
        self.assertEqual(captured_kwargs["env_config"], expected_env_config)
        self.assertEqual(captured_kwargs["agent_config"], agent_config["modules"])
        self.assertEqual(
            captured_kwargs["learner_seed"],
            runner_config["seed"]["learner"],
        )
        self.assertEqual(
            captured_kwargs["sampler_seed"],
            runner_config["seed"]["sampler"],
        )
        self.assertEqual(
            captured_kwargs["evaluator_seed"],
            runner_config["seed"]["evaluator"],
        )

    def test_environment_arguments_are_forwarded_opaquely(self):
        runner_config = {
            "env": {
                "module": "omnirl.envs.example",
                "game": "example",
                "custom_option": 42,
            }
        }
        agent_config = {"modules": {}}

        with patch.object(Runner, "__init__", return_value=None) as init:
            Runner.from_config(object(), object(), runner_config, agent_config)

        self.assertEqual(
            init.call_args.kwargs["env_config"],
            {"game": "example", "custom_option": 42},
        )


if __name__ == "__main__":
    unittest.main()
