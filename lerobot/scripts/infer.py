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
from lerobot.configs.infer import InferPipelineConfig, InferConfig

from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    predict_action
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect, busy_wait

from lerobot.common.datasets.utils import dataset_to_policy_features, get_features_from_robot
import cv2

normal=0
stopWhenFinish=1
pause=2
stopExit=3
stop_level: int =normal # 0= normal 1= stop when finish,  2= pause , 3 =stop and exit

# Global variable for image display
task_image: np.ndarray = np.zeros((240, 960, 3), dtype=np.uint8)

def infer_policy(robot: FairinoRobot, policy: PreTrainedPolicy, loop_time:int , fps: int, device, use_amp: bool) :
    listener, events = init_keyboard_listener()

    # Connect robot and set robot to init pos 
    if not robot.is_connected:
        robot.connect()
    # robot to initial pose
    joint_pos = robot.capture_observation()["observation.state"].numpy()
    while not np.allclose(joint_pos, robot.config.initial_pos, atol=2.0):
        time.sleep(0.01)
        joint_pos = robot.capture_observation()["observation.state"].numpy()
        clear_camera_buffer_count = clear_camera_buffer_count - 1

    loop_count = 0 
    maybe_loop_done_count = 0

    while   (loop_time < 0 or loop_count< loop_time) and   not events["exit_infer"] :
        if stop_level==3:
            break
        
        start_loop_t = time.perf_counter()
        observation = robot.capture_observation()
        joint_pos = observation["observation.state"].numpy()
        image_keys = [key for key in observation if "image" in key]
        x_offset= 0
        for key in image_keys:
                img = observation[key].numpy()                      
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #putting all together
                task_image[:,x_offset:x_offset+img.shape[1]//2] = img[:,0:img.shape[1]//2]
                x_offset += img.shape[1]//2
       
                    
        pred_action = predict_action(observation, policy, device, use_amp)

        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        # Caution ！！！ make sure the pred action is in reasonable range before send_action
        
        action = robot.send_action(pred_action)
        print(pred_action)
        
        if (np.allclose(pred_action.numpy(), robot.config.initial_pos, atol=5) and \
            np.allclose(joint_pos, robot.config.initial_pos, atol=5)) :
            maybe_loop_done_count += 1
        else :
            maybe_loop_done_count = 0
        if maybe_loop_done_count > fps * 5 :
            print("maybe inference task done")
            break
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
        loop_count = loop_count + 1
        # log the real infer fps pre 1 s
        if loop_count % fps == 0 :
            log_control_info(robot, dt_s, fps=fps)
    listener.stop()
        

@parser.wrap()
def FairinoRobotInfer(policy_path: str, loop_time:int=-1, policy: PreTrainedPolicy=None):

    cfg = InferConfig(policy_path)

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
    
    if policy is None:
        logging.info("Making policy.")
        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.policy.pretrained_path, 
                                        config=cfg.policy,
                                        local_files_only=True,
                                        map_location=cfg.device)
        
    logging.info("Start inference ....")
    infer_policy(robot=robot, policy=policy, loop_time=loop_time, fps=30, device=device, use_amp=cfg.use_amp)
    robot.disconnect()
    logging.info("End of infer")
    


if __name__ == "__main__":
    # Register our custom control config
    FairinoRobotInfer("outputs/train/test3/checkpoints/last/pretrained_model")
