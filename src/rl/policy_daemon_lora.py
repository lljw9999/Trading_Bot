"""
Policy Daemon with LoRA Delta Checkpoint Support
SAC-DiF RL policy training with PEFT (Parameter-Efficient Fine-Tuning)
"""

import os
import torch
import torch.nn as nn
import redis
import hashlib
import json
import numpy as np
from typing import Dict, Any
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import logging
import pynvml


class ActorModel(nn.Module):
    """Simple Actor model for SAC-DiF."""

    def __init__(self, state_dim: int = 32, action_dim: int = 1, net_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, action_dim * 2),  # mean and log_std
        )

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class PolicyDaemonWithLora:
    """
    Policy daemon with LoRA delta checkpoint support.

    Features:
    - LoRA configuration for parameter-efficient fine-tuning
    - Delta checkpoint saving (< 200kB vs full model)
    - Resume from delta checkpoints
    - Model hash tracking in Redis
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 1,
        net_dim: int = 256,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        """
        Initialize policy daemon with LoRA support.

        Args:
            state_dim: State vector dimension
            action_dim: Action vector dimension
            net_dim: Hidden layer dimension
            lora_r: LoRA rank parameter
            lora_alpha: LoRA scaling parameter
        """
        self.logger = logging.getLogger("policy_daemon_lora")
        self.r = redis.Redis(host="localhost", port=6379, decode_responses=True)

        # GPU Auto-allocator setup
        self._setup_gpu_allocator()

        # Initialize base actor model
        self.actor = ActorModel(state_dim, action_dim, net_dim)

        # LoRA configuration as specified in task brief
        if os.getenv("LORA"):  # we're resuming a delta
            self.logger.info(f"Loading LoRA delta from {os.getenv('LORA')}")
            base = torch.load("/models/base_actor.pth")
            self.actor.load_state_dict(base, strict=False)
            self.actor = get_peft_model(
                self.actor, LoraConfig(r=lora_r, lora_alpha=lora_alpha)
            )
            self.actor.load_state_dict(torch.load(os.getenv("LORA")))
        else:
            # Convert to LoRA model for training
            self.logger.info("Initializing new LoRA model")
            self.actor = get_peft_model(
                self.actor,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=[
                        "net.0",
                        "net.2",
                        "net.4",
                    ],  # Apply LoRA to linear layers
                    lora_dropout=0.1,
                ),
            )

        self.step_count = 0

        # Track model parameters for monitoring
        total_params = sum(p.numel() for p in self.actor.parameters())
        trainable_params = sum(
            p.numel() for p in self.actor.parameters() if p.requires_grad
        )

        self.logger.info(
            f"Model initialized - Total: {total_params:,} params, Trainable: {trainable_params:,} params"
        )
        self.logger.info(
            f"LoRA efficiency: {trainable_params/total_params:.1%} of parameters are trainable"
        )

    def _setup_gpu_allocator(self):
        """Set up GPU memory allocation based on available memory."""
        try:
            if not torch.cuda.is_available():
                self.logger.info("CUDA not available, skipping GPU allocator")
                return

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            free_mb = memory_info.free // (1024**2)
            reserve_mb = 1024  # Keep 1 GB headroom

            # Adaptive memory fraction based on available memory
            if free_mb > 8000:
                mem_fraction = 0.9  # Use 90% if > 8GB free
            else:
                mem_fraction = 0.75  # Use 75% if limited memory

            torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)

            self.logger.info(
                f"GPU allocator: {free_mb}MB free, using {mem_fraction:.0%} fraction"
            )

            # Store GPU metrics for Prometheus
            self.gpu_mem_frac = mem_fraction
            self.r.set("gpu:mem_frac", str(mem_fraction))

        except Exception as e:
            self.logger.warning(f"GPU allocator setup failed: {e}")
            self.gpu_mem_frac = 0.8  # Default fallback

    def save_delta_checkpoint(self, step: int) -> str:
        """
        Save LoRA delta checkpoint instead of full weights.

        Args:
            step: Training step number

        Returns:
            checkpoint_hash: SHA1 hash of the delta
        """
        try:
            # Get only the LoRA parameters (delta)
            delta = get_peft_model_state_dict(self.actor)

            # Save delta checkpoint
            delta_path = f"/models/delta/{step}.dlt"
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)
            torch.save(delta, delta_path)

            # Calculate hash of delta for tracking
            delta_bytes = b"".join([v.cpu().numpy().tobytes() for v in delta.values()])
            checkpoint_hash = hashlib.sha1(delta_bytes).hexdigest()

            # Store in Redis for system tracking
            self.r.set("model:hash", checkpoint_hash)
            self.r.hset(
                "model:info",
                mapping={
                    "step": step,
                    "delta_path": delta_path,
                    "hash": checkpoint_hash,
                    "size_kb": os.path.getsize(delta_path) / 1024,
                    "timestamp": str(torch.tensor(0).item()),  # Current timestamp
                },
            )

            self.logger.info(
                f"Saved delta checkpoint: {delta_path} ({os.path.getsize(delta_path)/1024:.1f}kB)"
            )
            self.logger.info(f"Model hash: {checkpoint_hash}")

            return checkpoint_hash

        except Exception as e:
            self.logger.error(f"Error saving delta checkpoint: {e}")
            return ""

    def load_delta_checkpoint(self, delta_path: str) -> bool:
        """Load LoRA delta checkpoint."""
        try:
            if not os.path.exists(delta_path):
                self.logger.error(f"Delta checkpoint not found: {delta_path}")
                return False

            delta = torch.load(delta_path, map_location="cpu")
            self.actor.load_state_dict(delta, strict=False)

            self.logger.info(f"Loaded delta checkpoint: {delta_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading delta checkpoint: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        try:
            model_info = self.r.hgetall("model:info")

            # Add current model statistics
            total_params = sum(p.numel() for p in self.actor.parameters())
            trainable_params = sum(
                p.numel() for p in self.actor.parameters() if p.requires_grad
            )

            return {
                **model_info,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "lora_efficiency": f"{trainable_params/total_params:.2%}",
                "step_count": self.step_count,
                "is_lora_model": True,
            }

        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {}

    def sigstore_sign_delta(self, delta_path: str) -> str:
        """
        Sigstore signing scaffold for model security.

        Args:
            delta_path: Path to delta checkpoint file

        Returns:
            signature_path: Path to signature file
        """
        try:
            step = os.path.splitext(os.path.basename(delta_path))[0]
            sig_path = f"/models/delta/{step}.sig"

            # Sigstore signing command (requires sigstore CLI installed)
            import subprocess

            cmd = f"sigstore sign {delta_path} --output {sig_path}"

            self.logger.info(f"Signing command (manual execution required): {cmd}")
            self.logger.info("Note: Install sigstore CLI: pip install sigstore")

            # For now, create a placeholder signature file
            with open(sig_path, "w") as f:
                f.write(f"# Sigstore signature placeholder for {delta_path}\n")
                f.write(f"# Execute: {cmd}\n")
                f.write(f"# Generated at step {step}\n")

            return sig_path

        except Exception as e:
            self.logger.error(f"Error in sigstore signing: {e}")
            return ""

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step (placeholder implementation).

        Args:
            batch_data: Training batch with states, actions, rewards, etc.

        Returns:
            training_metrics: Loss values and other metrics
        """
        self.step_count += 1

        # Placeholder training logic
        states = batch_data.get("states", torch.randn(32, 32))
        actions = batch_data.get("actions", torch.randn(32, 1))

        # Forward pass
        mean, log_std = self.actor(states)

        # Simple MSE loss (placeholder)
        loss = torch.nn.functional.mse_loss(mean, actions)

        # NaN/OOM guards and fail-fast
        import time

        if not torch.isfinite(loss).all():
            error_msg = (
                f"NaN/Inf detected in loss at step {self.step_count}: {loss.item()}"
            )
            self.logger.error(error_msg)
            # Publish alert to Redis
            self.r.lpush("alerts:policy", f"{int(time.time())}:NaN_LOSS:{error_msg}")
            raise RuntimeError(error_msg)

        try:
            # Backward pass (would include optimizer.step() in real implementation)
            loss.backward()

            # Check gradients for NaN/Inf
            for name, param in self.actor.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    error_msg = f"NaN/Inf gradient in {name} at step {self.step_count}"
                    self.logger.error(error_msg)
                    self.r.lpush(
                        "alerts:policy", f"{int(time.time())}:NaN_GRAD:{error_msg}"
                    )
                    raise RuntimeError(error_msg)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                error_msg = f"OOM detected at step {self.step_count}: {str(e)}"
                self.logger.error(error_msg)
                self.r.lpush("alerts:policy", f"{int(time.time())}:OOM:{error_msg}")
                raise RuntimeError(error_msg)
            else:
                raise

        # Update Redis heartbeat and stats every step
        current_ts = int(time.time())
        entropy = (
            torch.distributions.Normal(mean, log_std.exp()).entropy().mean().item()
        )

        # Update heartbeat with TTL
        self.r.setex("policy:last_update_ts", 48 * 3600, current_ts)  # 48h TTL

        # Update current stats
        self.r.hset(
            "policy:current",
            mapping={
                "entropy": entropy,
                "step": self.step_count,
                "loss": loss.item(),
                "timestamp": current_ts,
                "collapse_risk": (
                    "HIGH" if entropy < 0.3 else "MEDIUM" if entropy < 0.8 else "LOW"
                ),
            },
        )

        # Save checkpoint every 1000 steps
        checkpoint_hash = ""
        if self.step_count % 1000 == 0:
            checkpoint_hash = self.save_delta_checkpoint(self.step_count)

            # Sign the checkpoint
            delta_path = f"/models/delta/{self.step_count}.dlt"
            if os.path.exists(delta_path):
                sig_path = self.sigstore_sign_delta(delta_path)
                self.logger.info(f"Created signature placeholder: {sig_path}")

        return {
            "step": self.step_count,
            "actor_loss": loss.item(),
            "checkpoint_hash": checkpoint_hash,
            "lora_params": sum(
                p.numel() for p in self.actor.parameters() if p.requires_grad
            ),
        }

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action from current policy."""
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()
            action = torch.normal(mean, std)
            return torch.tanh(action)  # Bound action to [-1, 1]


def create_policy_daemon_with_lora(**kwargs) -> PolicyDaemonWithLora:
    """Factory function for creating LoRA policy daemon."""
    return PolicyDaemonWithLora(**kwargs)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("🚀 SAC-DiF Policy Daemon with LoRA Delta Checkpoints")

    # Initialize daemon
    daemon = create_policy_daemon_with_lora(
        state_dim=32, action_dim=1, net_dim=256, lora_r=8, lora_alpha=16
    )

    # Demonstrate training step
    batch_data = {"states": torch.randn(32, 32), "actions": torch.randn(32, 1)}

    metrics = daemon.train_step(batch_data)
    print(f"Training metrics: {metrics}")

    # Show model info
    info = daemon.get_model_info()
    print(f"Model info: {info}")

    # Demonstrate action inference
    test_state = torch.randn(1, 32)
    action = daemon.get_action(test_state)
    print(f"Sample action: {action.item():.4f}")
