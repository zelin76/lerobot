import argparse
import time

import numpy as np

from lerobot.common.robot_devices.robots.fr3_arm import FairinoArm
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
def test_robot(ip_address) :
    config = MotorsBusConfig(
        port=ip_address,
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
    fr3_robot.connect()
    fr3_robot.enable()
    count = 10000
    target_pos = np.array([-90., -90,  -90, -90, -90, -90. ])
    target_pos1 = np.array([0, 0, 0, 0, 0, 0])
    while count :
        count-=1
        
        joint_pos = fr3_robot.getJointPos()
        target_pos1 = target_pos1 * 0.9 + target_pos * 0.1
        fr3_robot.setJointPos(target_pos1)
        print(joint_pos)
        time.sleep(0.01)

    fr3_robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="FR3 Robot ip (e.g. 127.0.0.1)")
    args = parser.parse_args()

    test_robot(args.ip)