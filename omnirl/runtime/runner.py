import os
import time
from datetime import timedelta
from functools import partial
import threading

import ray

from .evaluator import Evaluator
from .file_writer import FileWriter
from .logger import setup_logger
from .sampler import Sampler
from .weight_store import WeightStore


class Runner:
    """
    Distributed training runner.

    Args:
        agent: Algorithm module that provides Learner, Policy, and Memory.
        environment: Environment backend module used for interaction and evaluation.
        agent_config: Configuration for agent modules.
        env_config: Configuration for environment modules.
        ckpt_dir: Directory path to save model checkpoints.
        eval_model: Checkpoint file name of the model used for evaluation.
        num_learners: Number of learner actors.
        num_samplers: Number of sampler actors.
        num_evaluators: Number of evaluator actors.
        num_episodes: Number of evaluation episodes per evaluator.
        max_updates: Maximum number of learner updates.
        save_interval: Time in seconds between checkpoint saves.
        log_interval: Time in seconds between logging events.
        log_name: Name of the training log file.
        load_checkpoint: Whether to resume training from an existing checkpoint.
        learner_device: Device selector used to run the learner.
        learner_seed: Random seed used for learner.
        sampler_seed: Random seed used for sampler.
        evaluator_seed: Random seed used for evaluator.
    """

    def __init__(
        self,
        agent,
        environment,
        agent_config,
        env_config,
        ckpt_dir: str = "checkpoints",
        eval_model: str | None = None,
        num_learners: int = 1,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        num_episodes: int = 10,
        max_updates: int = 10000000,
        save_interval: int = 3600,
        log_interval: int = 60,
        log_name: str = "log.csv",
        load_checkpoint: bool = True,
        learner_device: str | None = None,
        learner_seed: int | None = None,
        sampler_seed: int | None = None,
        evaluator_seed: int | None = None,
    ):
        # Initialize ray
        try:
            runtime_env = {"working_dir": ".", "excludes": ["checkpoints/"]}
            ray.init(address="auto", runtime_env=runtime_env)
        except ConnectionError:
            ray.init()

        # Load agent module arguments from config
        learner_kwargs = dict(agent_config.get("learner") or {})
        memory_kwargs = dict(agent_config.get("memory") or {})
        policy_kwargs = dict(agent_config.get("policy") or {})
        network_kwargs = dict(agent_config.get("network") or {})
        unroll_length = memory_kwargs.get("unroll_length", 1)

        # Preserve the complete environment configuration as opaque backend data
        env_config = dict(env_config)

        # Read the environment description through the backend interface
        env_spec = environment.get_env_spec(env_config)
        state_dim = env_spec.state_dim
        state_dtype = env_spec.state_dtype
        action_dim = env_spec.action_dim
        is_continuous = env_spec.is_continuous
        action_low = env_spec.action_low
        action_high = env_spec.action_high
        default_eval_metrics = dict(env_spec.default_eval_metrics)

        # Initialize learner
        num_gpus = 0 if learner_device == "cpu" else 1
        self.learners = [
            agent.Learner.options(num_gpus=num_gpus).remote(
                state_dim=state_dim,
                action_dim=action_dim,
                ckpt_dir=ckpt_dir,
                device=learner_device,
                is_continuous=is_continuous,
                seed=learner_seed,
                network_kwargs=network_kwargs,
                rank=i,
                world_size=num_learners,
                **learner_kwargs,
            )
            for i in range(num_learners)
        ]
        self.learner = self.learners[0]
        if eval_model is not None:
            ray.get(self.learner.load_model.remote(eval_model))

        # Initialize memory
        self.memory = agent.Memory.remote(
            state_dim=state_dim,
            action_dim=action_dim,
            state_dtype=state_dtype,
            is_continuous=is_continuous,
            **memory_kwargs,
        )

        # Initialize weight store
        init_weight = ray.get(self.learner.get_weights.remote())
        init_version = ray.get(self.learner.get_version.remote())
        self.weight_store = WeightStore.remote(init_weight, init_version)

        # Initialize samplers
        self.samplers = [
            Sampler.remote(
                policy_cls=agent.Policy,
                collection_cls=environment.CollectionLoop,
                env_builder=partial(
                    environment.build_env,
                    env_config=env_config,
                    role="collection",
                ),
                state_dim=state_dim,
                action_dim=action_dim,
                is_continuous=is_continuous,
                action_low=action_low,
                action_high=action_high,
                seed=None if sampler_seed is None else sampler_seed + 1000 + i,
                network_kwargs=network_kwargs,
                policy_kwargs=policy_kwargs,
                unroll_length=unroll_length,
            )
            for i in range(num_samplers)
        ]

        # Initialize evaluators
        self.evaluators = [
            Evaluator.remote(
                policy_cls=agent.Policy,
                evaluation_cls=environment.EvaluationLoop,
                env_builder=partial(
                    environment.build_env,
                    env_config=env_config,
                    role="evaluation",
                ),
                state_dim=state_dim,
                action_dim=action_dim,
                num_episodes=num_episodes,
                is_continuous=is_continuous,
                action_low=action_low,
                action_high=action_high,
                seed=None if evaluator_seed is None else evaluator_seed + 2000 + i,
                network_kwargs=network_kwargs,
                policy_kwargs=policy_kwargs,
            )
            for i in range(num_evaluators)
        ]

        # Initialize training configuration
        self.max_updates = max_updates
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.load_checkpoint = load_checkpoint

        # Initialize logging and checkpoint utilities
        self.log = setup_logger()
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        self.fw = FileWriter(ckpt_dir=self.ckpt_dir, log_name=log_name)

        # Initialize training state
        self.version = init_version
        self.eval_version = -1
        self.trained_samples = 0

        # Initialize cached statistics
        self.latest_train_info = {}
        self.latest_eval_info = {
            "version": 0,
            "num_episodes": 0,
            **default_eval_metrics,
        }

        # Initialize timing state
        self.start_time = 0.0
        self.last_log_time = 0.0
        self.next_log_time = float("inf")
        self.next_save_time = float("inf")
        self.stop_event = threading.Event()
        self.aggregate_evaluations = environment.aggregate_evaluations

        # Initialize throughput counters
        self.last_collected_data = 0
        self.last_trained_samples = 0

    @classmethod
    def from_config(cls, agent, environment, runner_config, agent_config):
        """Create a Runner instance from parsed configuration dictionaries."""
        device_param = runner_config.get("device", {})
        seed_param = runner_config.get("seed", {})
        env_param = dict(runner_config["env"])
        env_param.pop("module")

        # Load runner and environment arguments
        runner_kwargs = {
            "agent": agent,
            "environment": environment,
            "agent_config": agent_config["modules"],
            "env_config": env_param,
            **runner_config.get("runner", {}),
        }

        # Load device arguments
        device_mapping = {"learner": "learner_device"}
        for src_key, dst_key in device_mapping.items():
            if src_key in device_param:
                runner_kwargs[dst_key] = device_param[src_key]

        # Load seed arguments
        seed_mapping = {
            "learner": "learner_seed",
            "sampler": "sampler_seed",
            "evaluator": "evaluator_seed",
        }
        for src_key, dst_key in seed_mapping.items():
            if src_key in seed_param:
                runner_kwargs[dst_key] = seed_param[src_key]

        return cls(**runner_kwargs)

    def _log_str(self, string):
        """Log a centered banner string."""
        self.log.info("-" * 59)
        self.log.info(f"|{string:^57}|")
        self.log.info("-" * 59)

    def _log_groups(self, groups):
        """Format grouped metrics and write to logger."""
        lines = self._build_log_lines(groups)
        for line in lines:
            self.log.info(line)

    def _build_log_lines(self, groups):
        """Build aligned text lines for grouped metric logging."""
        key_width = 22
        val_width = 30
        total_width = key_width + val_width + 7

        lines = ["-" * total_width]

        for group_name, metrics in groups:
            if not metrics:
                continue

            lines.append(f"| {group_name:<{key_width + val_width + 3}} |")
            for key, value in metrics.items():
                value = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"|    {key:<{key_width - 3}} | {value:<{val_width}} |")

        lines.append("-" * total_width)
        return lines

    def _load_checkpoint(self):
        """Load learner checkpoint and synchronize weights."""
        if self.load_checkpoint and os.path.exists(self.ckpt_path):
            remote = [learner.load_checkpoint.remote() for learner in self.learners]
            results = ray.get(remote)[0]

            self.version = results["version"]
            weights = results["weights"]
            ray.get(self.weight_store.set.remote(weights, self.version))

            self._log_str("RESUME FROM CHECKPOINT")

    def _start_samplers(self):
        """Launch asynchronous rollout tasks on all sampler actors."""
        for sampler in self.samplers:
            sampler.rollout.remote(self.memory, self.weight_store)

    def _start_evaluators(self, is_train=True):
        """Run evaluators in parallel and aggregate evaluation metrics."""
        while not self.stop_event.is_set() or not is_train:
            results = ray.get(self.weight_store.get.remote(self.eval_version))
            version = results["version"]
            weights = results["weights"]

            if weights is None:
                self.stop_event.wait(1.0)
                continue

            eval_refs = []
            for evaluator in self.evaluators:
                eval_refs.append(evaluator.evaluate.remote(weights, version))
            eval_infos = ray.get(eval_refs)

            avg_info = self.aggregate_evaluations(eval_infos, version)

            self.latest_eval_info = avg_info
            self.eval_version = version

            if not is_train:
                break

        return avg_info

    def _train_one_step(self):
        """Run learner update if memory has enough data."""
        if not ray.get(self.memory.ready.remote()):
            return

        batch_refs = []
        for i in range(len(self.learners)):
            batch_refs.append(self.memory.get_batch.remote(i, len(self.learners)))

        lb_zip = zip(self.learners, batch_refs)
        learn_list = [learner.learn.remote(batch_ref) for learner, batch_ref in lb_zip]
        infos = ray.get(learn_list)

        train_info = infos[0]
        avg_log = {}
        for key in train_info["log"]:
            avg_log[key] = sum(info["log"][key] for info in infos) / len(infos)
        self.latest_train_info = avg_log

        self.version = train_info["version"]
        self.trained_samples += sum(info["trained_samples"] for info in infos)

        weights_ref = self.learner.get_weights.remote()
        ray.get(self.weight_store.set.remote(weights_ref, self.version))
        ray.get(self.memory.on_learn.remote([info["memory"] for info in infos]))

    def _log_info(self, force=False):
        """Collect runtime statistics, evaluate policy, and write training logs."""
        log_start = time.perf_counter()
        if log_start < self.next_log_time and not force:
            return

        collected_data = ray.get(self.memory.stats.remote())["collected_data"]

        dt = log_start - self.last_log_time
        wall_time = log_start - self.start_time

        sampler_fps = (collected_data - self.last_collected_data) / dt
        learner_fps = (self.trained_samples - self.last_trained_samples) / dt

        groups = [
            (
                "time/",
                {
                    "updates": f"{self.version} / {self.max_updates}",
                    "wall_time": str(timedelta(seconds=int(wall_time))),
                    "collected_samples": collected_data,
                    "trained_samples": self.trained_samples,
                    "sampler_fps": int(sampler_fps),
                    "learner_fps": int(learner_fps),
                },
            ),
            ("train/", self.latest_train_info),
            ("eval/", self.latest_eval_info),
        ]

        self._log_groups(groups)

        self.fw.write(
            {
                "wall_time": wall_time,
                "updates": self.version,
                "collected_samples": collected_data,
                "trained_samples": self.trained_samples,
                "sampler_fps": sampler_fps,
                "learner_fps": learner_fps,
                **self.latest_eval_info,
                **self.latest_train_info,
            }
        )

        self.last_log_time = log_start
        self.last_collected_data = collected_data
        self.last_trained_samples = self.trained_samples
        self.next_log_time = log_start + self.log_interval

    def _save_checkpoint(self):
        """Save learner checkpoint when the save interval is reached."""
        now = time.perf_counter()
        if now < self.next_save_time:
            return

        ray.get(self.learner.save_checkpoint.remote())
        self._log_str("SAVE CHECKPOINTS")
        self.next_save_time = now + self.save_interval

    def fit(self):
        """Run the full distributed training loop."""
        self._log_str("START TRAINING")
        self._load_checkpoint()
        self._start_samplers()

        self.stop_event.clear()
        eval_thread = threading.Thread(target=self._start_evaluators, daemon=True)
        eval_thread.start()

        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

        self.next_log_time = self.start_time + self.log_interval
        self.next_save_time = self.start_time + self.save_interval

        try:
            while self.version < self.max_updates:
                self._train_one_step()
                self._log_info()
                self._save_checkpoint()

            self._log_info(force=True)
            self._log_str("FINISH TRAINING")

        finally:
            self.stop_event.set()
            eval_thread.join()
            for sampler in self.samplers:
                ray.kill(sampler)
            ray.get(self.learner.save_checkpoint.remote())
            self.fw.close()

    def eval(self):
        """Run distributed evaluation on all evaluators and report results."""
        self._log_str("START EVALUATION")
        self.latest_eval_info = self._start_evaluators(is_train=False)
        groups = [("eval/", self.latest_eval_info)]
        self._log_groups(groups)
        self._log_str("FINISH EVALUATION")

        return groups
