import numpy as np


class EvaluationLoop:
    """
    Run evaluation episodes using the Procgen interaction protocol.

    Args:
        env_builder: Callable that builds a new environment instance.
        seed: Random seed used for the initial environment reset.
    """

    def __init__(self, env_builder, seed=None):
        self.env = env_builder()
        self.seed = seed

    def run(self, choose_action, num_episodes):
        """Evaluate an action callback for a fixed number of episodes."""
        plan = self.env.configure(self.seed, num_episodes)
        evaluations = []
        for assignment in plan["assignments"]:
            self.env.activate(assignment)
            self.env.reset(seed=self.seed)
            episode_returns = []

            for _ in range(assignment["episodes"]):
                observation, _ = self.env.reset()
                done = False
                episode_return = 0.0

                while not done:
                    if self.env.render_mode == "human":
                        self.env.render()

                    action = choose_action(observation)
                    observation, reward, terminated, truncated, _ = self.env.step(
                        action
                    )
                    done = terminated or truncated
                    episode_return += float(reward)

                episode_returns.append(episode_return)

            returns = np.asarray(episode_returns, dtype=np.float32)
            evaluations.append(
                {
                    "split": assignment["split"],
                    "num_episodes": int(assignment["episodes"]),
                    "metrics": {
                        "return_mean": float(returns.mean()),
                        "return_std": float(returns.std()),
                        "return_max": float(returns.max()),
                        "return_min": float(returns.min()),
                    },
                }
            )

        return {
            "worker_index": plan["worker_index"],
            "num_workers": plan["num_workers"],
            "evaluations": evaluations,
        }


def _aggregate_split(eval_infos):
    """Aggregate metrics for one Procgen evaluation split."""
    num_episodes = sum(info["num_episodes"] for info in eval_infos)
    return_mean = sum(info["metrics"]["return_mean"] for info in eval_infos)
    return_mean /= len(eval_infos)

    square_sum = 0.0
    for info in eval_infos:
        metrics = info["metrics"]
        square_sum += metrics["return_std"] ** 2 + metrics["return_mean"] ** 2
    mean_square = square_sum / len(eval_infos)
    return_std = (mean_square - return_mean**2) ** 0.5

    return {
        "num_episodes": int(num_episodes),
        "return_mean": float(return_mean),
        "return_std": float(return_std),
        "return_max": max(info["metrics"]["return_max"] for info in eval_infos),
        "return_min": min(info["metrics"]["return_min"] for info in eval_infos),
    }


def aggregate_evaluations(eval_infos, version):
    """Aggregate metrics returned by Procgen evaluation workers."""
    num_workers = {info["num_workers"] for info in eval_infos}
    worker_indices = {info["worker_index"] for info in eval_infos}
    if len(num_workers) != 1 or worker_indices != set(range(num_workers.pop())):
        raise ValueError(
            "runner.num_evaluators must match env.evaluation_num_evaluators."
        )

    evaluations = [
        evaluation for info in eval_infos for evaluation in info["evaluations"]
    ]
    train_infos = [info for info in evaluations if info["split"] == "train"]
    test_infos = [info for info in evaluations if info["split"] == "test"]

    result = {"version": version, "num_episodes": 0}
    if train_infos:
        train_metrics = _aggregate_split(train_infos)
        result["num_episodes"] = train_metrics.pop("num_episodes")
        result.update({f"train_{name}": value for name, value in train_metrics.items()})
    if test_infos:
        test_metrics = _aggregate_split(test_infos)
        test_metrics.pop("num_episodes")
        result.update({f"test_{name}": value for name, value in test_metrics.items()})
    return result
