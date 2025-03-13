"""
This script configure a single motor at a time to a given ID and baudrate.

Example of usage:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem585A0080521 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```
"""

import argparse
import time
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import (
    MODEL_BAUDRATE_TABLE,
    SCS_SERIES_BAUDRATE_TABLE,
    FeetechMotorsBus,
)
import numpy as np
import logging
from lerobot.common.robot_devices.robots.fr3_arm import FairinoArm
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.utils import busy_wait

leader_arm_offset = np.array([0, -90, 0, 0, 0, 0])
leader_arm_dir = np.array([1, 1, -1, -1, 1, 1])
follow_arm_limit_min = np.array([-170, -260, -155, -260, -170, -170])
follow_arm_limit_max = np.array([170, 80, 155, 80, 170, 170])

def ensure_safe_goal_position(
    goal_pos: np.ndarray, present_pos: np.ndarray, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    diff_ratio =  max_relative_target /  np.abs(diff)  
    min_ratio = np.min(diff_ratio)
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

def test_bus():
    config = MotorsBusConfig(
        port="/dev/ttyACM0",
        motors={
            # name: (index, model)
            "shoulder_pan": [1, "sts3215"],
            "shoulder_lift": [2, "sts3215"],
            "elbow_flex": [3, "sts3215"],
            "wrist_flex": [4, "sts3215"],
            "wrist_roll": [5, "sts3215"],
            "wrist_yaw": [6, "sts3215"],
        },
        mock=False
    )
    
    # Initialize the MotorBus with the correct port and motor configurations
    motor_bus = FeetechMotorsBus(config=config)
    config = MotorsBusConfig(
        port="192.168.58.2",
        motors={
            # name: (index, model)
            "shoulder_pan": [1, "fr3"],
            "shoulder_lift": [2, "fr3"],
            "elbow_flex": [3, "fr3"],
            "wrist_flex": [4, "fr3"],
            "wrist_roll": [5, "fr3"],
            "wrist_yaw": [6, "fr3"],
        },
        mock=False
    )
    fr3_robot = FairinoArm(config)
    
    
    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        fr3_robot.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return
    motor_bus.write("Torque_Enable", 0)
    fr3_robot.enable()
    # Motor bus is connected, proceed with the rest of the operations
    try:
        while 1:
            start_loop_t = time.perf_counter()
            leadr_arm_position = motor_bus.read("Present_Position")
            leadr_arm_position = (leadr_arm_position - 2048) * 0.08789 * leader_arm_dir + leader_arm_offset
            
            follow_arm_position = fr3_robot.getJointPos()
            print("leader Present Position", leadr_arm_position)
            print("follow Present Position", follow_arm_position)
            leadr_arm_position = leadr_arm_position.clip(min=follow_arm_limit_min, max=follow_arm_limit_max)
            safe_goal = ensure_safe_goal_position(leadr_arm_position, follow_arm_position, 2)
            #if not np.allclose(follow_arm_position, safe_goal, atol=1):
            fr3_robot.setJointPos(safe_goal, cmd_T=0.02)
            print("safe goal      Position", safe_goal)
            dt_s = time.perf_counter() - start_loop_t
            print("dt:", dt_s)
            busy_wait(0.02 - dt_s)

            time.sleep(0.02)

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":

    test_bus()
