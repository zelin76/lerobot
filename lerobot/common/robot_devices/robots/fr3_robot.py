"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    MotorsBusConfig,
)

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id, ensure_safe_goal_position

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

from lerobot.common.robot_devices.motors.feetech import TorqueMode

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.robots.fr3_arm import FairinoArm

@dataclass
class Fr3obotConfig :
    # initial pos
    initial_pos = np.array([-53.9997, -116.9998, 96.9997,  -106.9997, 7.0001, -120.9999, 0, \
                           -39.9998, -63.9998,  -77.9992, -86.9995,  1.9999, 126.5697,  0,  0])
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target         = np.array([5,5, 5, 5, 5, 5, 10])
    
    ##主从手臂关节的映射关系 方向、偏置、 比例
    leader_arm_encoder2deg      = 180.0 / 2048.0
    leader_arm_encoder_offset   = 2048

    leader_arm_deg_offset : dict[str, np.array] = field(
        default_factory=lambda: { 
            "left" : np.array([-45, -90, 0, -90, 0, 45]),
            "right": np.array([-45, -90, 0, -90, 0, -45])
        }
    )
    leader_arm_dir : dict[str, np.array] = field(
        default_factory=lambda: {
            "left" : np.array([1, 1, -1, -1, 1, -1]),
            "right": np.array([1,  1, -1, -1, 1, -1])
        }
    )
    follow_arm_limit_min        = np.array([-170, -150, -155, -260, -170, -170])
    follow_arm_limit_max        = np.array([170, 80, 155, 80, 170, 170])

    ##夹爪的编码器范围 完全闭合到完全张开的范围 用于将夹爪位置映射到 0~100
    leader_gripper_encoder_range = [2048, 2900] 
    follow_gripper_encoder_range : dict[str, list] = field(
        default_factory=lambda: {
            "left" : [2048, 2900],
            "right": [2048, 2900]
        }
    )
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": MotorsBusConfig(
                serial_port="/dev/ttyACM4",#"/dev/leader_arm_left",
                ip_address=None,
                motors={
                    # name: (index, model)
                    "shoulder_pan":  [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex":    [3, "sts3215"],
                    "wrist_flex":    [4, "sts3215"],
                    "wrist_roll":    [5, "sts3215"],
                    "wrist_yaw":     [6, "sts3215"],
                    "gripper":       [7, "sts3215"],
                    "head_yaw":      [8, "sts3215"]
                },
            ),
            "right": MotorsBusConfig(
                serial_port="/dev/ttyACM3",#"/dev/leader_arm_right",
                ip_address=None,
                motors={
                    # name: (index, model)
                    "shoulder_pan":  [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex":    [3, "sts3215"],
                    "wrist_flex":    [4, "sts3215"],
                    "wrist_roll":    [5, "sts3215"],
                    "wrist_yaw":     [6, "sts3215"],
                    "gripper":       [7, "sts3215"]
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": MotorsBusConfig(
                serial_port="/dev/gripper_left",
                ip_address="192.168.57.3",
                motors={
                    # name: (index, model)
                    "shoulder_pan":  [1, "fr3"],
                    "shoulder_lift": [2, "fr3"],
                    "elbow_flex":    [3, "fr3"],
                    "wrist_flex":    [4, "fr3"],
                    "wrist_roll":    [5, "fr3"],
                    "wrist_yaw":     [6, "fr3"],
                    "gripper":       [7, "sts3215"]
                },
            ),
            "right": MotorsBusConfig(
                serial_port="/dev/gripper_right",
                ip_address="192.168.57.2",
                motors={
                    # name: (index, model)
                    "shoulder_pan":  [1, "fr3"],
                    "shoulder_lift": [2, "fr3"],
                    "elbow_flex":    [3, "fr3"],
                    "wrist_flex":    [4, "fr3"],
                    "wrist_roll":    [5, "fr3"],
                    "wrist_yaw":     [6, "fr3"],
                    "gripper":       [7, "sts3215"]
                },
            ),
        }
    )
    head_config = MotorsBusConfig(
        serial_port="/dev/head_yaw",
        ip_address=None,
        motors={
            # name: (index, model)
            "head":  [1, "sts3215"]
        },
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left": OpenCVCameraConfig(
                camera_index=0,
                camera_port="/dev/camera_left",
                fps=30,
                width=640,
                height=240,
            ),
            "right": OpenCVCameraConfig(
                camera_index=1,
                camera_port="/dev/camera_right",
                fps=30,
                width=640,
                height=240,
            ),
            "head": OpenCVCameraConfig(
                camera_index=1,
                camera_port="/dev/camera_head",
                fps=30,
                width=640,
                height=240,
            ),
        }
    )


class FairinoRobot:

    def __init__(
        self,
        teleop_mode: bool,
        left_com:str="COM7", #leader arm com
        right_com:str="COM9" #leader arm com
    ):
        self.config = Fr3obotConfig()
        self.robot_type = "fairino_robot"
        #leader arm
        self.leader_arms : dict[str, FeetechMotorsBus] = {}
        if teleop_mode :
            self.config.leader_arms["left"].serial_port=left_com
            self.config.leader_arms["right"].serial_port=right_com
            
            for key, cfg in self.config.leader_arms.items():
                self.leader_arms[key] = FeetechMotorsBus(cfg)
        #follow arm 
        self.follower_arms : dict[str, FairinoArm] = {}
        self.cameras = {}
        self.head_motor = None
        if not teleop_mode:
            for key, cfg in self.config.follower_arms.items():
                self.follower_arms[key] = FairinoArm(cfg, self.config.follow_gripper_encoder_range[key])
            self.head_motor = FeetechMotorsBus(self.config.head_config)
            #cameras
            self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        motor_name : list = [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]
        motor_name.append("head_yaw")
        return motor_name

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.follower_arms)
        state_names = self.get_motor_names(self.follower_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )
        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].enable()

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].getJointPos()

        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")
        
        if self.head_motor :
            print("connect head motor")
            self.head_motor.connect()
            # Mode=0 for Position Control
            self.head_motor.write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.head_motor.write("P_Coefficient", 8)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.head_motor.write("I_Coefficient", 0)
            self.head_motor.write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.head_motor.write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.head_motor.write("Maximum_Acceleration", 20)
            self.head_motor.write("Acceleration", 20)

            self.head_motor.write("Torque_Enable", TorqueMode.ENABLED.value)
            self.head_motor.read("Present_Position")
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
            

        self.is_connected = True
        print("fr3 robot init done...")

    def align_gripper_position(self, gripper_position) : #map gripper position to 0~100
        align_position = (gripper_position - self.config.leader_gripper_encoder_range[0]) * 100 \
                         / (self.config.leader_gripper_encoder_range[1] - self.config.leader_gripper_encoder_range[0])
        align_position = np.clip(align_position, 0, 100)
        return align_position
    
    def head_motor_encoder2position(self, head_position):
        align_position = (head_position - self.config.leader_arm_encoder_offset) * \
                    self.config.leader_arm_encoder2deg
        return align_position
    
    def head_motor_position2encoder(self, head_position):
        align_position = head_position / self.config.leader_arm_encoder2deg + self.config.leader_arm_encoder_offset
        return align_position
    
    def align_position(self, key, origin_position) :
        leadr_arm_position = origin_position[:-1]
        gripper_position =  self.align_gripper_position(origin_position[-1])
        leadr_arm_position = (leadr_arm_position - self.config.leader_arm_encoder_offset) * \
                            self.config.leader_arm_encoder2deg * self.config.leader_arm_dir[key] + self.config.leader_arm_deg_offset[key]
        
        leadr_arm_position = leadr_arm_position.clip(min=self.config.follow_arm_limit_min, \
                                                     max=self.config.follow_arm_limit_max)
        return np.append(leadr_arm_position, gripper_position)
    
    def get_leader_pos(self) -> dict[str, np.ndarray] :
        leader_pos = {}
        head_motor_pos : np.ndarray
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            if name == "left" :
                head_motor_pos = self.head_motor_encoder2position(leader_pos[name][ 7 : 8]) * -1
                leader_pos[name] = leader_pos[name][ : -1]
            leader_pos[name] = self.align_position(name, leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
        leader_pos["head"] = head_motor_pos
        return leader_pos
    
    def teleop_step(
        self, record_data=False, DT=0.02
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        # Prepare to assign the position of the leader to the follower
        leader_pos = self.get_leader_pos()
        for name in leader_pos:
            leader_pos[name] = torch.from_numpy(leader_pos[name])
        #print("leader ", leader_pos)
        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]
           
            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].getJointPos()
                #print("present_pos ", present_pos)
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos
            goal_pos = goal_pos.numpy().astype(np.int32)
            
            self.follower_arms[name].setJointPos(goal_pos, cmd_T=DT)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t
        head_pos = leader_pos["head"].numpy().astype(np.int32)
        head_pos = self.head_motor_position2encoder(head_pos)
        head_pos = head_pos.astype(np.int32)
        self.head_motor.write("Goal_Position", self.head_motor_position2encoder(head_pos))
        # Early exit when recording data is not requested
        if not record_data:
            return

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action.append(head_pos)
        action = torch.cat(action)

        # Populate output dictionnaries
        action_dict = {}
        action_dict["action"] = action
       
        return self.capture_observation(), action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].getJointPos()
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t
        if self.head_motor :
            follower_pos["head"] = self.head_motor_encoder2position(self.head_motor.read("Present_Position"))
            follower_pos["head"] = torch.from_numpy(follower_pos["head"])
        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state.append(follower_pos["head"])
        state = torch.cat(state)
        state = state.type(torch.float32)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor, DT=0.02) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        start_frame_time1 = time.perf_counter()
        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motors)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            
            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                start_frame_time = time.perf_counter()
                present_pos = self.follower_arms[name].getJointPos().astype(np.float32)
                dt_s = time.perf_counter() - start_frame_time
                #print("333fr3 elapsed time:", dt_s*1000,"ms")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)
            # Save tensor to concat and return
            action_sent.append(goal_pos)
            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            start_frame_time = time.perf_counter()
            self.follower_arms[name].setJointPos(goal_pos, cmd_T=DT)
            dt_s = time.perf_counter() - start_frame_time
            #print("444fr3 elapsed time:", dt_s*1000,"ms")
        head_pos = action[from_idx : from_idx+1]
        action_sent.append(head_pos)
        head_pos = self.head_motor_position2encoder(head_pos)
        head_pos = head_pos.numpy().astype(np.int32)
        start_frame_time = time.perf_counter()
        self.head_motor.write("Goal_Position", head_pos)
        dt_s = time.perf_counter() - start_frame_time
        #print("555fr3 elapsed time:", dt_s*1000,"ms")
        dt_s = time.perf_counter() - start_frame_time1
        #print("fr3 total elapsed time:", dt_s*1000,"ms")
        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disable()
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        
        for name in self.cameras:
            self.cameras[name].disconnect()

        if self.head_motor :
            self.head_motor.disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
