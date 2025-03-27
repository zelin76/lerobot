import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs.types import FeatureType
from lerobot.common.policies.factory import  get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.infer import InferPipelineConfig

from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    predict_action
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect, busy_wait

from lerobot.common.datasets.utils import dataset_to_policy_features, get_features_from_robot

def infer_policy(robot: FairinoRobot, policy: PreTrainedPolicy, fps: int, device, use_amp: bool) :
    listener, events = init_keyboard_listener()

    # Connect robot and set robot to init pos 
    if not robot.is_connected:
        robot.connect()
    # TODO get the robot home pose
    # robot.toInitPos()

    loop_count = 0
    while  not events["exit_infer"] :
        start_loop_t = time.perf_counter()
        observation = robot.capture_observation()
        pred_action = predict_action(observation, policy, device, use_amp)

        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        
        # action = robot.send_action(pred_action)
        print(pred_action)
        
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
        loop_count = loop_count + 1
        # log the real infer fps pre 1 s
        if loop_count % fps == 0 :
            log_control_info(robot, dt_s, fps=fps)
    listener.stop()
        

@parser.wrap()
def FairinoRobotInfer(cfg: InferPipelineConfig):
    """Main server control function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making Robot.")
    robot = FairinoRobot(teleop_mode=False)
    
    logging.info("Making policy.")
   
    policy_cls = get_policy_class(cfg.policy.type)

    policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.policy.pretrained_path, 
                                        config=cfg.policy,
                                        local_files_only=True,
                                        map_location=cfg.device)

    logging.info("Start inference ....")
    infer_policy(robot=robot, policy=policy, fps=30, device=device, use_amp=cfg.use_amp)
    robot.disconnect()
    logging.info("End of infer")

if __name__ == "__main__":
    # Register our custom control config
    FairinoRobotInfer()