"""
Server script for controlling follower arms and cameras in a robot teleoperation setup.

This script connects only to the follower arms and cameras from the setup in fr3_robot.py.
It listens for commands sent from control_robot_client.py via socket communication.

Examples of usage:

- Start server to record data from follower arms controlled by leader arms:
```bash
python lerobot/scripts/control_robot_server.py --control.type=record --control.listen_port=9999 --control.fps=30 --control.repo_id=fr3/test3 --control.num_episodes=2 --control.single_task="Grasp a lego block and put it in the bin."
```
"""

import logging
import time
import socket
import pickle
import argparse
import threading
from dataclasses import asdict, dataclass, field
from pprint import pformat

import torch
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser


@dataclass
class ServerControlConfig(RecordControlConfig):
    """Configuration for server control mode."""
    listen_port: int = 9999  # Default port to listen on


class FollowerRobot(FairinoRobot):
    """Modified FairinoRobot that only connects follower arms and cameras."""
    
    def __init__(self, teleop_mode: bool = False):
        # Initialize with the parent class config but customize components
        super().__init__(teleop_mode=False)
        self.robot_type = "fairino_follower_robot"
        
        # Clear leader arms since this robot only uses follower arms
        self.leader_arms = {}
        
        # Follower arms and cameras are inherited from parent class
        # No need to reinitialize them as they're already set up in super().__init__
        
        self.is_connected = False
        self.logs = {}


class SocketServer:
    """Socket server to receive commands from the client."""
    
    def __init__(self, port=9999, buffer_size=4096):
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.client_socket = None
        self.should_run = False
        self.latest_action = None
        self.action_lock = threading.Lock()
    
    def start(self):
        """Start the socket server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(1)
        logging.info(f"Server listening on port {self.port}")
        
        self.should_run = True
        self.receiver_thread = threading.Thread(target=self._receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
    
    def _receive_data(self):
        """Receive data from the client in a separate thread."""
        logging.info("Waiting for client connection...")
        self.client_socket, addr = self.socket.accept()
        logging.info(f"Client connected from {addr}")
        
        while self.should_run:
            try:
                data = self.client_socket.recv(self.buffer_size)
                if not data:
                    logging.info("Client disconnected")
                    break
                
                # Deserialize the received action data
                action_data = pickle.loads(data)
                
                # Update the latest action
                with self.action_lock:
                    self.latest_action = action_data
                
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                break
        
        if self.client_socket:
            self.client_socket.close()
    
    def get_latest_action(self):
        """Get the latest action received from the client."""
        with self.action_lock:
            return self.latest_action
    
    def stop(self):
        """Stop the socket server."""
        self.should_run = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()


@safe_disconnect
def server_record(
    robot: Robot,
    cfg: ServerControlConfig,
) -> LeRobotDataset:
    """Record data from follower arms controlled by leader arms via socket."""
    # Setup socket server
    server = SocketServer(port=cfg.listen_port)
    server.start()
    
    # Handle dataset resume/new creation
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            local_files_only=cfg.local_files_only,
        )
        # Initialize image writer (multi-process/thread)
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        # Validate dataset compatibility with robot
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create new dataset
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy (if provided)
    policy = None if cfg.policy is None else make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()
    # Initialize keyboard listener (for controlling recording flow)
    listener, events = init_keyboard_listener()
    
    # Wait for client to connect and provide first action
    log_say("Waiting for client connection", cfg.play_sounds)
    while server.get_latest_action() is None:
        time.sleep(0.1)
    
    # Warmup phase before recording (adjust starting position/device sync)
    log_say("Starting warmup phase", cfg.play_sounds)
    enable_teleoperation = False  # Don't enable teleoperation as we're receiving actions from the client
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)
    log_say("Ready to record", cfg.play_sounds)
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()  # Safety stop check

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        # Start recording a single episode
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        
        # Custom record episode function for server receiving actions from client
        server_record_episode(
            dataset=dataset,
            robot=robot,
            server=server,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_cameras=cfg.display_cameras,
            fps=cfg.fps,
        )

        # Environment reset phase
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Resetting environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s)

        # Handle re-recording logic
        if events["rerecord_episode"]:
            log_say("Re-recording current episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()  # Clear current episode cache
            continue

        # Save current episode
        dataset.save_episode(cfg.single_task)
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    # Post-recording cleanup
    log_say("Stopping recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_cameras)
    server.stop()

    # Calculate dataset statistics (optional)
    if cfg.run_compute_stats:
        logging.info("Computing dataset statistics")

    dataset.consolidate(cfg.run_compute_stats)

    log_say("Exiting program", cfg.play_sounds)
    return dataset


def server_record_episode(
    dataset,
    robot,
    server,
    events,
    episode_time_s,
    display_cameras=True,
    fps=30,
):
    """Record a single episode using actions received from the client via socket."""
    # Initialize episode timing
    episode_frame_limit = None if episode_time_s is None else int(episode_time_s * fps)
    episode_frame_count = 0
    
    # Main recording loop
    dt_window = []
    while True:
        start_frame_t = time.perf_counter()
        
        # Get the latest action from the server
        action_data = server.get_latest_action()
        if action_data is None:
            time.sleep(0.01)  # Small sleep to prevent CPU hammering
            continue
        
        # Convert action data to tensor
        action = torch.tensor(action_data, dtype=torch.float32)
        
        # Send action to robot and get observation
        robot.send_action(action, DT=1.0/fps)
        obs_dict = robot.capture_observation()
        
        # Add frame to dataset
        action_dict = {"action": action}
        dataset.add_frame(obs_dict, action_dict)
        
        # Track timing for FPS control
        dt_s = time.perf_counter() - start_frame_t
        dt_window.append(dt_s)
        if len(dt_window) > 100:
            dt_window.pop(0)
        avg_dt_s = sum(dt_window) / len(dt_window)
        current_fps = 1.0 / avg_dt_s if avg_dt_s > 0 else float('inf')
        
        # Display control info and camera feeds
        log_control_info(robot, dt_s, fps=fps)
        if display_cameras:
            for camera_name, image in obs_dict.items():
                if camera_name.startswith("observation.images."):
                    camera_key = camera_name.split(".")[-1]
                    robot.cameras[camera_key].async_show(image)
        
        # Check for early exit conditions
        if events["exit_early"] or events["stop_recording"] or events["rerecord_episode"]:
            break
        
        # Check if we've reached the maximum episode frames
        episode_frame_count += 1
        if episode_frame_limit is not None and episode_frame_count >= episode_frame_limit:
            break
        
        # Wait to maintain target FPS
        sleep_time = max(0, 1.0 / fps - dt_s)
        busy_wait(sleep_time)


@parser.wrap()
def control_robot_server(cfg: ControlPipelineConfig):
    """Main server control function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Create a robot instance that only connects follower arms and cameras
    robot = FollowerRobot(teleop_mode=False)

    # Execute the server record mode
    if isinstance(cfg.control, ServerControlConfig):
        server_record(robot, cfg.control)
    else:
        raise ValueError(f"Unsupported control type: {type(cfg.control)}")

    # Safely disconnect
    if robot.is_connected:
        robot.disconnect()


if __name__ == "__main__":
    # Replace RecordControlConfig with ServerControlConfig in parser
    parser.add_config("control", ServerControlConfig)
    control_robot_server()
