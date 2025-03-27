#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
from pathlib import Path

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed

from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs.policies import PreTrainedConfig
import os
from torch.optim import Adam,Optimizer
from torch.optim.lr_scheduler import StepLR

def load_pretrained_model(model_path: str, dataset_path: str):
    """Load pretrained model from specified path.
    
    Args:
        model_path: Path to model checkpoint directory
        dataset_path: Dataset repository ID for model configuration
    
    Returns:
        tuple: (step, policy, optimizer, lr_scheduler) loaded from checkpoint
    """
    from lerobot.common.utils.train_utils_plus import load_checkpoint
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig
    
    # Create config with actual dataset metadata
    policyPath = model_path+ "/pretrained_model"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset_path),
        policy=PreTrainedConfig.from_pretrained(policyPath)
        #device=get_safe_torch_device()
    )
    
    # Load actual dataset to get observation space
    #temp_dataset = make_dataset(cfg)
    #cfg.policy.observation_space = temp_dataset.observation_space
    
    #cfg.validate()

    # Load checkpoint with proper dataset metadata and device initialization
    step, policy, states = load_checkpoint(
        model_path,
        
    )
    
    # Initialize optimizer with proper parameter groups
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(
    #    cfg.optim,
    #    [{"params": policy.parameters()}],
    #    optimizer_state=states.get("optimizer"),
    #    lr_scheduler_state=states.get("lr_scheduler")
    #)
    
    optimizer = Adam(policy.parameters())
    lr_scheduler = StepLR(optimizer, step_size=1000)
    step=0

    return (
        step,
        policy.to(cfg.device),
        optimizer,
        lr_scheduler
    )
