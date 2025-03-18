import logging
import math
import time
import traceback
import numpy as np

from lerobot.common.robot_devices.fairino import Robot
from lerobot.common.robot_devices.motors.configs import MotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus, TorqueMode
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class FairinoArm:

    def __init__(
        self,
        config: MotorsBusConfig,
        gripper_encoder_range : list[int]
    ):
        self.ip_address = config.ip_address
        self.motors = config.motors
        self.arm = None
        self.is_connected = False
        self.gripper_encoder_range = gripper_encoder_range
        gripper_config = MotorsBusConfig(
            serial_port=config.serial_port,
            ip_address=None,
            motors={
            # name: (index, model)
            "gripper": [1, "sts3215"]}
        )
        self.gripper = FeetechMotorsBus(gripper_config)

    def set_gripper_preset(self):
        # Mode=0 for Position Control
        self.gripper.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        self.gripper.write("P_Coefficient", 16)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.gripper.write("I_Coefficient", 0)
        self.gripper.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.gripper.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.gripper.write("Maximum_Acceleration", 254)
        self.gripper.write("Acceleration", 254)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FairinoArm({self.ip_address}) is already connected. Do not call `FairinoArm.connect()` twice."
            )

        try:
            self.arm = Robot.RPC(self.ip_address)
            self.gripper.connect()
            if not self.arm.is_conect:
                raise OSError(f"Failed to connect arm '{self.ip_address}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\ncheck fr3 arm is connected and set pc ip address in the same .\n"
            )
            raise
        self.set_gripper_preset()
        # Allow to read and write
        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. Try running `FairinoArm.connect()` first."
            )
        self.gripper.disconnect()
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
        self.gripper.write("Torque_Enable", TorqueMode.ENABLED.value)
        self.arm.RobotEnable(1)
        self.arm.ServoMoveStart()
    
    def disable(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        self.gripper.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.arm.RobotEnable(0)
    def gripper_encoder2pos(self, endcoder_pos):
        pos = ((endcoder_pos - self.gripper_encoder_range[0]) * 100 
                / (self.gripper_encoder_range[1] - self.gripper_encoder_range[0]))
        pos = np.clip(pos, a_min=0, a_max=100)
        return pos
    def gripper_pos2encoder(self, pos):
        encoder_pos = int((pos / 100.0)*(self.gripper_encoder_range[1]-self.gripper_encoder_range[0])+self.gripper_encoder_range[0])
        #encoder_pos = np.clip(encoder_pos, a_min=self.gripper_encoder_range[0], a_max=self.gripper_encoder_range[1])
        #print("gripper", pos, encoder_pos)
        return encoder_pos
    
    def setJointPos(self, values: np.ndarray, cmd_T=0.02):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        gripper_values= values[-1]
        values = values[:-1]
        values = values.tolist()
        self.arm.ResetAllError()
        gripper_values = self.gripper_pos2encoder(gripper_values)
        self.gripper.write("Goal_Position", gripper_values)
        ret =  self.arm.ServoJ(joint_pos=values, axisPos=[0,0,0,0,0,0], cmdT=cmd_T)
        return ret
    
    def getJointPos(self) :
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FairinoArm({self.ip_address}) is not connected. You need to run `FairinoArm.connect()`."
            )
        err, joint_pos = self.arm.GetActualJointPosDegree()
        gripper_pos = self.gripper_encoder2pos(self.gripper.read("Present_Position"))
        joint_pos = np.array(joint_pos)
        joint_pos = np.append(joint_pos, gripper_pos)
        return joint_pos