import logging
import math
import time
import traceback
import numpy as np

from lerobot.common.robot_devices.fairino import Robot
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class FairinoArm:

    def __init__(
        self,
        config: MotorsBusConfig,
    ):
        self.ip_address = config.port
        self.motors = config.motors
        self.arm = None
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FairinoArm({self.ip_address}) is already connected. Do not call `FairinoArm.connect()` twice."
            )

        try:
            self.arm = Robot.RPC(self.ip_address)
            if not self.arm.is_conect:
                raise OSError(f"Failed to connect arm '{self.ip_address}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\ncheck fr3 arm is connected and set pc ip address in the same .\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. Try running `FairinoArm.connect()` first."
            )
        self.arm.ServoMoveEnd()
        self.arm.CloseRPC()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def enable(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        self.arm.RobotEnable(1)
        self.arm.ServoMoveStart()
    
    def disable(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        self.arm.RobotEnable(0)

    def setJointPos(self,  values: np.ndarray, cmd_T=0.02):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        values = values.tolist()
        self.arm.ResetAllError()
        ret =  self.arm.ServoJ(joint_pos=values, axisPos=[0,0,0,0,0,0], cmdT=cmd_T)
        return ret
    
    def getJointPos(self) :
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        err, joint_pos = self.arm.GetActualJointPosDegree()
        joint_pos = np.array(joint_pos)

        return joint_pos