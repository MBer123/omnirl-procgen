import os

import ray
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

from omnirl.networks import ImpalaCNN
from omnirl.utils.seed import set_global_seed


@ray.remote
class Learner:
    """
    PPO learner agent.

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
        is_distributed = world_size > 1

        if is_distributed and not torch_dist.is_initialized():
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

        if is_distributed:
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

    def _evaluate_action(self, states_tensor, actions_tensor):
        """Evaluate action log probabilities and entropies under the current policy."""
        action_logits, state_values = self.train_model(states_tensor)
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

        # Compute GAE
        with torch.no_grad():
            flat_states = states_tensor.view(B * T, *states_tensor.shape[2:])
            flat_next_states = next_states_tensor.view(B * T, *states_tensor.shape[2:])

            state_values = self.model.forward_critic(flat_states).squeeze(-1)
            next_state_values = self.model.forward_critic(flat_next_states).squeeze(-1)

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
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_states = states_tensor.view(B * T, *states_tensor.shape[2:])
        flat_actions = actions_tensor.reshape(B * T)
        flat_old_log_probs = log_probs_tensor.reshape(B * T)
        flat_advantages = advantages.reshape(B * T)
        flat_returns = returns.reshape(B * T)

        N = B * T
        num_minibatches = min(self.num_minibatches, N)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_explained_var = 0.0
        num_updates = 0

        # PPO mini-batch learning loop
        for _ in range(self.k_epochs):
            indices = torch.randperm(N, device=self.device)
            minibatch_indices = torch.tensor_split(indices, num_minibatches)

            for mb_idx in minibatch_indices:
                if mb_idx.numel() == 0:
                    continue

                mb_states = flat_states[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                results = self._evaluate_action(mb_states, mb_actions)
                new_log_probs = results["log_probs"]
                dist_entropy = results["dist_entropy"]
                new_state_values = results["state_values"]

                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                ratio_max = 1 + self.epsilon_clip
                ratio_min = 1 - self.epsilon_clip
                cliped_ratios = torch.clamp(ratios, ratio_min, ratio_max)

                surrogate_1 = ratios * mb_advantages
                surrogate_2 = cliped_ratios * mb_advantages

                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                critic_loss = F.mse_loss(new_state_values, mb_returns)
                entropy_loss = dist_entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss
                loss = loss - self.ent_coef * entropy_loss

                approx_kl = (mb_old_log_probs - new_log_probs).mean()
                clip_frac = (torch.abs(ratios - 1.0) > self.epsilon_clip).float().mean()

                value_error_var = torch.var(mb_returns - new_state_values.detach())
                return_var = torch.var(mb_returns)
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
