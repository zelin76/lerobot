from typing import Protocol

from lerobot.common.robot_devices.robots.configs import (
    ManipulatorRobotConfig,
    RobotConfig,
)

import logging
import numpy as np
import torch

def ensure_safe_goal_position(
    goal_pos: np.ndarray, present_pos: np.ndarray, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    diff_ratio =  max_relative_target /  torch.abs(diff)  
    min_ratio = torch.min(diff_ratio)
    if min_ratio < 1:
        safe_diff = diff * min_ratio
    else:
        safe_diff = diff
    safe_goal_pos = present_pos + safe_diff

    # if not np.allclose(goal_pos, safe_goal_pos):
    #     logging.warning(
    #         "Relative goal position magnitude had to be clamped to be safe.\n"
    #         f"  requested relative goal position target: {diff}\n"
    #         f"    clamped relative goal position target: {safe_diff}"
    #     )

    return safe_goal_pos

def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


