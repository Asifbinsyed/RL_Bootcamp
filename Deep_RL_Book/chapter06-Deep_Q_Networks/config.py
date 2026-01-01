"""
Configuration file for DQN Pong training.
Adjust these parameters for local testing (M1 Pro) vs cloud GPU training.
This file is folder-agnostic and can be imported from anywhere.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DQNConfig:
    """DQN training configuration"""
    
    # Environment settings
    env_name: str = "ALE/Pong-v5"
    mean_reward_bound: float = 19.0  # Stop training when mean reward reaches this
    
    # Hyperparameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    
    # Buffer and batch settings
    # For local testing: smaller values for faster startup
    # For cloud GPU: increase REPLAY_SIZE to 50000-100000, REPLAY_START_SIZE to 10000
    replay_size: int = 5000  # Reduced for M1 Pro testing
    replay_start_size: int = 1000  # Reduced for M1 Pro testing (was 10000)
    batch_size: int = 16  # Reduced for M1 Pro CPU (was 32)
    
    # Target network sync
    sync_target_frames: int = 1000  # Frequency of syncing target network
    
    # Epsilon decay schedule
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay_last_frame: int = 20000  # Reduced for M1 Pro testing (was 150000)
    
    # Training limits (useful for local testing)
    max_frames: Optional[int] = None  # Set to limit total frames for testing
    max_time_minutes: Optional[int] = 15  # Stop after N minutes for local testing
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda" or "mps" (MPS for M1 Mac)
    
    # Logging
    tensorboard_comment: Optional[str] = None  # Auto-generated if None
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Auto-detect MPS for M1 Mac if device is cpu
        if self.device == "cpu":
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("MPS (Metal) detected - using MPS device for M1 Mac")
        
        # Generate tensorboard comment if not provided
        if self.tensorboard_comment is None:
            if self.max_frames or self.max_time_minutes:
                suffix = "test" if self.max_time_minutes else "local"
                self.tensorboard_comment = f"-{self.env_name}-{suffix}"
            else:
                self.tensorboard_comment = f"-{self.env_name}"


# Configuration presets
def get_local_config() -> DQNConfig:
    """Configuration optimized for local M1 Pro testing (10-15 minutes)"""
    return DQNConfig(
        replay_size=5000,
        replay_start_size=1000,
        batch_size=16,
        epsilon_decay_last_frame=20000,
        max_time_minutes=15,
        device="cpu",  # Will auto-detect MPS
    )


def get_cloud_config() -> DQNConfig:
    """Configuration optimized for cloud GPU training (full training, no time limit)"""
    return DQNConfig(
        replay_size=50000,
        replay_start_size=10000,
        batch_size=32,
        epsilon_decay_last_frame=150000,
        max_frames=None,
        max_time_minutes=None,
        device="cuda",
    )


def get_cloud_test_config() -> DQNConfig:
    """Configuration for cloud GPU testing (full cloud params but with time limit)"""
    return DQNConfig(
        replay_size=50000,
        replay_start_size=10000,
        batch_size=32,
        epsilon_decay_last_frame=150000,
        max_frames=None,
        max_time_minutes=30,  # 30 minutes test run on cloud
        device="cuda",
    )


def get_config(mode: str = "local") -> DQNConfig:
    """
    Get configuration preset.
    
    Args:
        mode: 
            - "local" for M1 Pro testing (small params, 15 min limit)
            - "cloud" for full GPU training (no time limit)
            - "cloud_test" for cloud GPU testing (full params, 30 min limit)
            - "custom" for manual config
        
    Returns:
        DQNConfig instance
    """
    if mode == "local":
        return get_local_config()
    elif mode == "cloud":
        return get_cloud_config()
    elif mode == "cloud_test":
        return get_cloud_test_config()
    else:
        # Return default config (can be customized)
        return DQNConfig()

