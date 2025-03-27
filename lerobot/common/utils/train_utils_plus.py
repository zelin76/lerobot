from typing import Tuple, Dict, Any
from pathlib import Path

from torch.optim import Adam,Optimizer
from torch.optim.lr_scheduler import StepLR

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.utils.train_utils import (
    load_training_state,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR
)
from lerobot.common.policies.act.configuration_act import ACTConfig

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import load_rng_state, save_rng_state
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig

from lerobot.common.policies.act.modeling_act import ACTPolicy


def load_checkpoint(checkout_dir: str) -> Tuple[int, PreTrainedPolicy, Dict[str, Any]]:
    """Load a training checkpoint from the specified directory.
    
    Args:
        checkout_dir: Path to directory containing the checkpoint
        
    Returns:
        Tuple containing:
        - Current training step
        - Loaded policy model
        - Dictionary containing optimizer and scheduler states
    """
    checkpoint_dir = Path(checkout_dir)
    
    # Load pretrained model
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    config=PreTrainedConfig.from_pretrained(pretrained_name_or_path=pretrained_dir)

    if config.type=="act":
        policy = ACTPolicy.from_pretrained(pretrained_name_or_path= pretrained_dir,config=config,local_files_only=True)
    
        # Create new optimizer and scheduler
    #optimizer =  Optimizer(config.get_optimizer_preset() )   
    optimizer = Adam(policy.parameters())
    scheduler = StepLR(optimizer, step_size=1000)
    step=0
    # Load training state into new objects
    #step, optimizer, scheduler = load_training_state(
    #    checkpoint_dir,
    #    optimizer
    #)
    
    return step, policy, {
        'optimizer': optimizer,
        'scheduler': scheduler
    }
