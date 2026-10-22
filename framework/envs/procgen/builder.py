import gym

from .adapter import EnvAdapter


class EvaluationEnv:
    """Lazily build the Procgen split assigned to an evaluator worker."""

    def __init__(self, config):
        self.config = dict(config)
        self.render_mode = self.config["render_mode"]
        self.env = None

    def configure(self, seed, _requested_episodes):
        """Reproduce the original train/test evaluator allocation."""
        total_levels = self.config["test_num_levels"]
        num_evaluators = self.config["evaluation_num_evaluators"]
        if num_evaluators <= 0:
            raise ValueError("evaluation_num_evaluators must be positive.")
        worker_index = (0 if seed is None else int(seed)) % num_evaluators

        if self.render_mode == "human":
            train_evaluators = num_evaluators
            test_evaluators = 0
        else:
            train_evaluators = max(1, (num_evaluators + 1) // 2)
            test_evaluators = max(1, num_evaluators // 2)

        assignments = []
        if worker_index < train_evaluators:
            assignments.append(
                self._assignment("train", worker_index, train_evaluators)
            )
        if self.render_mode != "human":
            if num_evaluators == 1:
                assignments.append(self._assignment("test", 0, test_evaluators))
            elif worker_index >= train_evaluators:
                assignments.append(
                    self._assignment(
                        "test",
                        worker_index - train_evaluators,
                        test_evaluators,
                    )
                )

        return {
            "worker_index": worker_index,
            "num_workers": num_evaluators,
            "assignments": assignments,
        }

    def _assignment(self, split, split_index, split_workers):
        """Describe one original Procgen evaluator assignment."""
        episodes = self.config["test_num_levels"] // split_workers
        base_level = (
            self.config["train_start_level"]
            if split == "train"
            else self.config["test_start_level"]
        )
        start_level = base_level + split_index * episodes
        return {
            "split": split,
            "episodes": episodes,
            "start_level": start_level,
            "render_mode": self.render_mode if split == "train" else None,
            "sequential_reset": self.render_mode != "human",
        }

    def activate(self, assignment):
        """Build the concrete environment for one assignment."""
        if self.env is not None:
            self.env.close()
        self.env = _build_adapter(
            self.config,
            render_mode=assignment["render_mode"],
            start_level=assignment["start_level"],
            num_levels=assignment["episodes"],
            sequential_reset=assignment["sequential_reset"],
        )

    def reset(self, seed=None, options=None):
        """Reset the assigned split environment."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Execute one environment step."""
        return self.env.step(action)

    def render(self):
        """Render the assigned split environment."""
        return self.env.render()

    def close(self):
        """Close the assigned split environment."""
        if self.env is not None:
            return self.env.close()
        return None


def _parse_config(env_config):
    """Parse the opaque Procgen environment configuration."""
    config = dict(env_config)
    try:
        env_id = config.pop("id")
    except KeyError:
        raise ValueError(
            "Procgen environment configuration requires an 'id'."
        ) from None

    parsed = {
        "id": env_id,
        "render_mode": config.pop("render_mode", None),
        "distribution_mode": config.pop("distribution_mode", "hard"),
        "num_stack": config.pop("num_stack", 4),
        "train_start_level": config.pop("train_start_level", 0),
        "train_num_levels": config.pop("train_num_levels", 500),
        "test_start_level": config.pop("test_start_level", 1000),
        "test_num_levels": config.pop("test_num_levels", 100),
        "evaluation_num_evaluators": config.pop("evaluation_num_evaluators", 10),
    }

    if config:
        fields = ", ".join(sorted(config))
        raise ValueError(f"Unsupported Procgen environment fields: {fields}.")
    return parsed


def _build_adapter(
    config,
    render_mode,
    start_level,
    num_levels,
    sequential_reset,
):
    """Build one concrete Procgen environment adapter."""
    env = gym.make(
        config["id"],
        render_mode=render_mode,
        distribution_mode=config["distribution_mode"],
        start_level=start_level,
        num_levels=num_levels,
    )
    return EnvAdapter(
        env=env,
        env_id=config["id"],
        render_mode=render_mode,
        distribution_mode=config["distribution_mode"],
        start_level=start_level,
        num_levels=num_levels,
        sequential_reset=sequential_reset,
        num_stack=config["num_stack"],
    )


def build_env(env_config, role=None):
    """Build and adapt a Procgen environment for a runtime role."""
    if role not in {None, "collection", "evaluation", "specification"}:
        raise ValueError(f"Unsupported environment role: {role!r}.")

    config = _parse_config(env_config)
    if role == "evaluation":
        return EvaluationEnv(config)

    return _build_adapter(
        config,
        render_mode=None,
        start_level=config["train_start_level"],
        num_levels=config["train_num_levels"],
        sequential_reset=False,
    )
