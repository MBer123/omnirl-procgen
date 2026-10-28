import os

import ray
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
from torch.nn.parallel import DistributedDataParallel as DDP

from framework.networks import ImpalaCNN
from omnirl.utils.seed import set_global_seed


class ValueNormalizer:
    """
    Running mean and variance tracker for unnormalized value targets.

    Args:
        epsilon: Initial sample count used to stabilize running statistics.
        device: Torch device used to store normalization statistics.
    """

    def __init__(self, epsilon: float = 1e-4, device: torch.device | None = None):
        self.mean = torch.zeros((), dtype=torch.float32, device=device)
        self.var = torch.ones((), dtype=torch.float32, device=device)
        self.count = torch.as_tensor(epsilon, dtype=torch.float32, device=device)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Merge batch moments into the running statistics."""
        batch_mean = torch.as_tensor(
            batch_mean, dtype=torch.float32, device=self.mean.device
        )
        batch_var = torch.as_tensor(
            batch_var, dtype=torch.float32, device=self.mean.device
        ).clamp_min(0.0)
        batch_count = torch.as_tensor(
            batch_count, dtype=torch.float32, device=self.mean.device
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = (m_2 / total_count).clamp_min(1e-12)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, values):
        """Map raw returns to the critic's target scale."""
        return (values - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, values):
        """Map critic predictions back to the reward scale."""
        return values * torch.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self):
        """Return a CPU copy of normalization state."""
        return {
            "mean": self.mean.detach().cpu().clone(),
            "var": self.var.detach().cpu().clone(),
            "count": self.count.detach().cpu().clone(),
        }

    def load_state_dict(self, state_dict):
        """Load normalization state."""
        self.mean = torch.as_tensor(
            state_dict["mean"], dtype=torch.float32, device=self.mean.device
        )
        self.var = torch.as_tensor(
            state_dict["var"], dtype=torch.float32, device=self.mean.device
        )
        self.count = torch.as_tensor(
            state_dict["count"], dtype=torch.float32, device=self.mean.device
        )


@ray.remote
class Learner:
    """
    SPPO learner: PPO with actor and critic augmentation plus ValueNorm.

    Args:
        state_dim: Dimension of the environment state space.
        action_dim: Dimension of the environment action space.
        is_continuous: Whether the action space is continuous.
        network_kwargs: Keyword arguments used to build the network.
        ckpt_dir: Directory path to save model checkpoints.
        gamma: Discount factor.
        lambda_: GAE lambda used for generalized advantage estimation.
        ent_coef: Entropy regularization coefficient used to encourage exploration.
        vf_coef: Coefficient for the critic loss in the total optimization objective.
        k_epochs: Number of optimization epochs performed for each rollout batch.
        num_minibatches: Number of minibatches per epoch.
        epsilon_clip: Clipping range used in the PPO surrogate objective.
        lr: Learning rate for the model optimizer.
        max_grad_norm: Maximum gradient norm used for gradient clipping.
        random_shift_padding: Edge padding used before shifting back to original size.
        drac_coef: Coefficient for policy KL regularization.
        device: Torch device to run the networks on.
        seed: Random seed used to ensure consistent results across runs.
        rank: Distributed learner rank.
        world_size: Total number of distributed learners.
        master_addr: Master address used to initialize distributed training.
        master_port: Master port used to initialize distributed training.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        is_continuous: bool = False,
        network_kwargs: dict | None = None,
        ckpt_dir: str = "checkpoints",
        gamma: float = 0.99,
        lambda_: float = 0.95,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        k_epochs: int = 3,
        num_minibatches: int = 8,
        epsilon_clip: float = 0.1,
        lr: float = 0.0003,
        max_grad_norm: float = 0.5,
        random_shift_padding: int = 4,
        drac_coef: float = 0.1,
        device: str | None = None,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ):
        # Limit PyTorch CPU threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Initialize random seed
        if seed is not None:
            set_global_seed(seed)

        # Create checkpoint directory
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Initialize device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        # Initialize distributed training
        self.rank = rank
        self.is_distributed = world_size > 1

        if self.is_distributed and not torch_dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            torch_dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=self.rank,
                world_size=world_size,
            )

        # Initialize parameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.num_minibatches = num_minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.random_shift_padding = random_shift_padding
        self.drac_coef = drac_coef
        self.value_norm = ValueNormalizer(device=self.device)

        # Initialize model network
        self.version = 0
        network_kwargs = dict(network_kwargs or {})
        model = ImpalaCNN(
            state_dim=state_dim,
            action_dim=action_dim,
            **network_kwargs,
        )

        self.model = model.to(self.device)
        self.model.train()

        if self.is_distributed:
            ddp_kwargs = {"device_ids": [0]} if self.device.type == "cuda" else {}
            self.train_model = DDP(self.model, **ddp_kwargs)
        else:
            self.train_model = self.model

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _state_dict(self, model):
        """Return a detached CPU copy of model state_dict."""
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def get_weights(self):
        """Export model network weights."""
        return self._state_dict(self.model)

    def get_version(self):
        """Export the current learner model version."""
        return self.version

    def _update_value_norm(self, returns):
        """Update return statistics and synchronize them across learners."""
        flat_returns = returns.reshape(-1).detach().float()

        if self.is_distributed:
            stats = torch.stack(
                [
                    flat_returns.sum(),
                    flat_returns.pow(2).sum(),
                    torch.as_tensor(
                        flat_returns.numel(), dtype=torch.float32, device=self.device
                    ),
                ]
            )
            torch_dist.all_reduce(stats, op=torch_dist.ReduceOp.SUM)
            count = stats[2].clamp_min(1.0)
            batch_mean = stats[0] / count
            batch_mean_sq = stats[1] / count
        else:
            count = torch.as_tensor(
                flat_returns.numel(), dtype=torch.float32, device=self.device
            ).clamp_min(1.0)
            batch_mean = flat_returns.mean()
            batch_mean_sq = flat_returns.pow(2).mean()

        batch_var = batch_mean_sq - batch_mean.pow(2)
        self.value_norm.update_from_moments(batch_mean, batch_var, count)

    def _random_shift(self, obs):
        """Apply random shift augmentation to NHWC image observations."""
        padding = self.random_shift_padding
        if padding <= 0 or obs.dim() != 4:
            return obs

        n, h, w, _ = obs.shape
        x = obs.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, (padding, padding, padding, padding), mode="replicate")

        shift_range = 2 * padding + 1
        offsets = torch.randint(shift_range * shift_range, (n,), device=obs.device)
        top = offsets // shift_range
        left = offsets % shift_range

        patches = x.unfold(2, h, 1).unfold(3, w, 1)
        batch_idx = torch.arange(n, device=obs.device)
        shifted = patches[batch_idx, :, top, left]
        return shifted.permute(0, 2, 3, 1).contiguous()

    def _drac_regularization(self, raw_output, aug_output):
        """Regularize augmented policy predictions toward raw predictions."""
        raw_dist = Categorical(logits=raw_output[0].detach())
        aug_dist = Categorical(logits=aug_output[0])
        return kl_divergence(raw_dist, aug_dist).mean()

    def _evaluate_model_output(self, model_output, actions_tensor):
        """Evaluate action log probabilities and entropies from model output."""
        action_logits, state_values = model_output
        dist = Categorical(logits=action_logits)

        dist_entropy = dist.entropy()
        log_probs = dist.log_prob(actions_tensor)

        return {
            "log_probs": log_probs.view(-1),
            "dist_entropy": dist_entropy.view(-1),
            "state_values": state_values.view(-1),
        }

    def learn(self, batch):
        """Sample a batch from rollout buffer and update the model network."""
        states_tensor = torch.tensor(batch["states"], device=self.device)
        actions_tensor = torch.tensor(batch["actions"], device=self.device)
        rewards_tensor = torch.tensor(batch["rewards"], device=self.device)
        next_states_tensor = torch.tensor(batch["next_states"], device=self.device)
        terminals_tensor = torch.tensor(batch["terminals"], device=self.device)
        dones_tensor = torch.tensor(batch["dones"], device=self.device)
        log_probs_tensor = torch.tensor(batch["log_probs"], device=self.device)

        states_tensor = states_tensor.float()
        next_states_tensor = next_states_tensor.float()
        terminals_tensor = terminals_tensor.float()
        dones_tensor = dones_tensor.float()

        B, T = rewards_tensor.shape

        with torch.no_grad():
            raw_reward_mean = rewards_tensor.mean().item()
            raw_reward_std = rewards_tensor.std(unbiased=False).item()

        # Compute GAE in raw reward units using the previous value statistics.
        with torch.no_grad():
            flat_states = states_tensor.view(B * T, *states_tensor.shape[2:])
            flat_next_states = next_states_tensor.view(B * T, *states_tensor.shape[2:])

            state_values = self.value_norm.denormalize(
                self.model.forward_critic(flat_states).squeeze(-1)
            )
            next_state_values = self.value_norm.denormalize(
                self.model.forward_critic(flat_next_states).squeeze(-1)
            )

            state_values = state_values.view(B, T)
            next_state_values = next_state_values.view(B, T)

            next_state_values = next_state_values * (1.0 - terminals_tensor)
            td_target = rewards_tensor + self.gamma * next_state_values
            deltas = td_target - state_values

            advantages = torch.zeros_like(rewards_tensor, device=self.device)
            gae = torch.zeros(B, device=self.device)

            for t in reversed(range(T)):
                gamma_lambda = self.gamma * self.lambda_ * (1.0 - dones_tensor[:, t])
                gae = deltas[:, t] + gamma_lambda * gae
                advantages[:, t] = gae

            returns = advantages + state_values
            self._update_value_norm(returns)
            normalized_returns = self.value_norm.normalize(returns)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            value_running_mean = self.value_norm.mean.item()
            value_running_std = torch.sqrt(self.value_norm.var + 1e-8).item()

        flat_states = states_tensor.view(B * T, *states_tensor.shape[2:])
        flat_actions = actions_tensor.reshape(B * T)
        flat_old_log_probs = log_probs_tensor.reshape(B * T)
        flat_advantages = advantages.reshape(B * T)
        flat_returns = returns.reshape(B * T)
        flat_normalized_returns = normalized_returns.reshape(B * T)

        N = B * T
        num_minibatches = min(self.num_minibatches, N)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_explained_var = 0.0
        total_policy_reg = 0.0
        num_updates = 0

        # PPO mini-batch learning loop
        for _ in range(self.k_epochs):
            indices = torch.randperm(N, device=self.device)
            minibatch_indices = torch.tensor_split(indices, num_minibatches)

            for mb_idx in minibatch_indices:
                if mb_idx.numel() == 0:
                    continue

                mb_states = flat_states[mb_idx]
                mb_aug_states = self._random_shift(mb_states)
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_normalized_returns = flat_normalized_returns[mb_idx]

                model_output = self.train_model(mb_states)
                aug_output = self.train_model(mb_aug_states)

                results = self._evaluate_model_output(model_output, mb_actions)
                new_log_probs = results["log_probs"]
                dist_entropy = results["dist_entropy"]
                new_state_values = results["state_values"]

                aug_results = self._evaluate_model_output(aug_output, mb_actions)
                aug_log_probs = aug_results["log_probs"]
                aug_state_values = aug_results["state_values"]

                ratio_max = 1 + self.epsilon_clip
                ratio_min = 1 - self.epsilon_clip

                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                aug_ratios = torch.exp(aug_log_probs - mb_old_log_probs)

                cliped_ratios = torch.clamp(ratios, ratio_min, ratio_max)
                aug_cliped_ratios = torch.clamp(aug_ratios, ratio_min, ratio_max)

                surrogate_1 = ratios * mb_advantages
                surrogate_2 = cliped_ratios * mb_advantages
                raw_actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

                surrogate_1 = aug_ratios * mb_advantages
                surrogate_2 = aug_cliped_ratios * mb_advantages
                aug_actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                actor_loss = 0.5 * raw_actor_loss + 0.5 * aug_actor_loss

                raw_critic_loss = F.mse_loss(new_state_values, mb_normalized_returns)
                aug_critic_loss = F.mse_loss(aug_state_values, mb_normalized_returns)
                critic_loss = 0.5 * raw_critic_loss + 0.5 * aug_critic_loss

                entropy_loss = dist_entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss
                loss = loss - self.ent_coef * entropy_loss

                policy_reg = self._drac_regularization(model_output, aug_output)
                loss = loss + self.drac_coef * policy_reg

                approx_kl = (mb_old_log_probs - new_log_probs).mean()
                clip_frac = (torch.abs(ratios - 1.0) > self.epsilon_clip).float().mean()

                raw_predicted_values = self.value_norm.denormalize(
                    new_state_values.detach()
                )
                value_error_var = torch.var(
                    mb_returns - raw_predicted_values, unbiased=False
                )
                return_var = torch.var(mb_returns, unbiased=False)
                explained_var = 1 - value_error_var / (return_var + 1e-8)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                total_explained_var += explained_var.item()
                total_policy_reg += policy_reg.item()
                num_updates += 1

        # Update model version
        self.version += 1

        return {
            "log": {
                "actor_loss": total_actor_loss / num_updates,
                "critic_loss": total_critic_loss / num_updates,
                "entropy": total_entropy / num_updates,
                "approx_kl": total_approx_kl / num_updates,
                "clip_frac": total_clip_frac / num_updates,
                "explained_var": total_explained_var / num_updates,
                "policy_reg": total_policy_reg / num_updates,
                "raw_reward_mean": raw_reward_mean,
                "raw_reward_std": raw_reward_std,
                "value_running_mean": value_running_mean,
                "value_running_std": value_running_std,
            },
            "memory": {"version": self.version},
            "version": self.get_version(),
            "trained_samples": int(B * T),
        }

    def save_checkpoint(self):
        """Save model and optimizer into checkpoint file."""
        if self.rank != 0:
            return

        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")

        ckpt = {
            "model": self._state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "value_norm": self.value_norm.state_dict(),
            "version": self.version,
        }
        torch.save(ckpt, path)

        eval_path = os.path.join(self.ckpt_dir, f"agent_{self.version}.pth")
        torch.save(self._state_dict(self.model), eval_path)

    def load_checkpoint(self):
        """Load model and optimizer from checkpoint file."""
        path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        ckpt = torch.load(path, map_location=self.device)

        # Load model network
        self.model.load_state_dict(ckpt["model"])

        # Load optimizer state
        self.optimizer.load_state_dict(ckpt["optimizer"])

        # Load value normalization state
        self.value_norm.load_state_dict(ckpt["value_norm"])

        # Load model version
        self.version = ckpt["version"]

        return {"version": self.get_version(), "weights": self.get_weights()}

    def load_model(self, model_path):
        """Load model from pre-trained model weights."""
        path = os.path.join(self.ckpt_dir, model_path)
        state_dict = torch.load(path, map_location=self.device)

        # Load model network
        self.model.load_state_dict(state_dict)

        # Load model version
        self.version = int(model_path.removeprefix("agent_").removesuffix(".pth"))
