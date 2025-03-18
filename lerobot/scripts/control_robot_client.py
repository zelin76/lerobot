"""
Client script for controlling leader arms in a robot teleoperation setup.

This script connects only to the leader arms from the setup in fr3_robot.py.
It sends the translated target positions to control_robot_server.py via socket communication.

Examples of usage:

- Start client to control leader arms and send commands to server:
```bash
python lerobot/scripts/control_robot_client.py --control.type=record --control.listen_ip=10.0.0.24 --control.listen_port=9999 --control.fps=30 --control.repo_id=fr3/test3 --control.num_episodes=2 --control.single_task="Grasp a lego block and put it in the bin."
```
"""

import logging
import time
import socket
import pickle
import threading
from dataclasses import asdict, dataclass, field
from pprint import pformat

import torch
import numpy as np

from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    stop_recording,
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser


@dataclass
class ClientControlConfig(RecordControlConfig):
    """Configuration for client control mode."""
    listen_ip: str = "127.0.0.1"  # IP address of the server to connect to
    listen_port: int = 9999  # Port to connect to on the server


class LeaderRobot(FairinoRobot):
    """Modified FairinoRobot that only connects leader arms."""
    
    def __init__(self, teleop_mode: bool = True):
        # Initialize with the parent class config but customize components
        super().__init__(teleop_mode=True)
        self.robot_type = "fairino_leader_robot"
        
        # Keep leader arms from parent initialization
        # Clear follower arms and cameras
        self.follower_arms = {}
        self.cameras = {}
        
        # These are already set in super().__init__ but let's be explicit
        self.is_connected = False
        self.logs = {}
    
    def get_aligned_leader_positions(self):
        """Get the aligned positions of the leader arms for the follower arms."""
        if not self.is_connected:
            raise ValueError("Robot is not connected")
        
        leader_pos = {}
        for name in self.leader_arms:
            # Read position from leader arm
            pos = self.leader_arms[name].read("Present_Position")
            # Align position for follower arm
            leader_pos[name] = self.align_position(name, pos)
        
        return leader_pos


class SocketClient:
    """Socket client to send commands to the server."""
    
    def __init__(self, server_ip="127.0.0.1", server_port=9999, buffer_size=4096):
        self.server_ip = server_ip
        self.server_port = server_port
        self.buffer_size = buffer_size
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.connected = True
            logging.info(f"Connected to server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to server: {e}")
            self.connected = False
            return False
    
    def send_data(self, data):
        """Send data to the server."""
        if not self.connected:
            return False
        
        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            # Send the data
            self.socket.sendall(serialized_data)
            return True
        except Exception as e:
            logging.error(f"Failed to send data: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            self.socket.close()
        self.connected = False


@safe_disconnect
def client_record(
    robot: LeaderRobot,
    cfg: ClientControlConfig,
):
    """Record leader arm positions and send them to the server."""
    # Setup socket client
    client = SocketClient(server_ip=cfg.listen_ip, server_port=cfg.listen_port)
    
    if not robot.is_connected:
        robot.connect()
    
    # Initialize keyboard listener
    listener, events = init_keyboard_listener()
    
    # Connect to the server
    log_say("Connecting to server...", cfg.play_sounds)
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        if client.connect():
            break
        retry_count += 1
        log_say(f"Connection attempt {retry_count} failed. Retrying...", cfg.play_sounds)
        time.sleep(1)
    
    if not client.connected:
        log_say("Failed to connect to server. Exiting.", cfg.play_sounds)
        return
    
    log_say("Connected to server. Starting recording session.", cfg.play_sounds)
    
    # Main control loop
    recorded_episodes = 0
    while recorded_episodes < cfg.num_episodes:
        # Start recording episode
        log_say(f"Recording episode {recorded_episodes + 1}", cfg.play_sounds)
        
        # Record episode timing
        episode_start_time = time.time()
        episode_frame_count = 0
        episode_frame_limit = None if cfg.episode_time_s is None else int(cfg.episode_time_s * cfg.fps)
        
        # Episode recording loop
        while True:
            start_frame_t = time.perf_counter()
            
            # Get aligned leader positions for follower arms
            leader_positions = robot.get_aligned_leader_positions()
            
            # Concatenate positions from all arms
            all_positions = []
            for name in leader_positions:
                all_positions.append(leader_positions[name])
            
            if all_positions:
                # Convert to numpy array
                all_positions = np.concatenate(all_positions)
                
                # Send positions to server
                if not client.send_data(all_positions):
                    log_say("Connection to server lost", cfg.play_sounds)
                    break
            
            # Track timing for FPS control
            dt_s = time.perf_counter() - start_frame_t
            
            # Display control info
            log_control_info(robot, dt_s, fps=cfg.fps)
            
            # Check for early exit conditions
            if events["exit_early"] or events["stop_recording"] or events["rerecord_episode"]:
                break
            
            # Check if we've reached the maximum episode frames
            episode_frame_count += 1
            if episode_frame_limit is not None and episode_frame_count >= episode_frame_limit:
                break
            
            # Wait to maintain target FPS
            sleep_time = max(0, 1.0 / cfg.fps - dt_s)
            busy_wait(sleep_time)
        
        # Handle re-recording logic
        if events["rerecord_episode"]:
            log_say("Re-recording current episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            continue
        
        # Episode completed
        recorded_episodes += 1
        
        # Reset environment period
        if recorded_episodes < cfg.num_episodes and not events["stop_recording"]:
            log_say("Reset environment for next episode", cfg.play_sounds)
            reset_start_time = time.time()
            
            while time.time() - reset_start_time < cfg.reset_time_s:
                if events["exit_early"] or events["stop_recording"]:
                    break
                time.sleep(0.1)
        
        if events["stop_recording"]:
            break
    
    # Cleanup
    log_say("Recording completed", cfg.play_sounds)
    client.disconnect()
    stop_recording(robot, listener, False)


@parser.wrap()
def control_robot_client(cfg: ControlPipelineConfig):
    """Main client control function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Create a robot instance that only connects leader arms
    robot = LeaderRobot(teleop_mode=True)

    # Execute the client record mode
    if isinstance(cfg.control, ClientControlConfig):
        client_record(robot, cfg.control)
    else:
        raise ValueError(f"Unsupported control type: {type(cfg.control)}")

    # Safely disconnect
    if robot.is_connected:
        robot.disconnect()


if __name__ == "__main__":
    # Replace RecordControlConfig with ClientControlConfig in parser
    parser.add_config("control", ClientControlConfig)
    control_robot_client()
