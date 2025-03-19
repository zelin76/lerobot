"""
Server utility for the FR3 robot control system.

This script connects to the follower_arms and cameras of the FR3 robot,
receives joint positions data from the client, and controls the follower_arms
accordingly. It also records all data including the received leader_arms data.

Example usage:
```bash
python lerobot/scripts/control_robot_server.py \
    --control.type=server_record \
    --control.listen_port=9999 \
    --control.fps=30 \
    --control.repo_id=fr3/test3 \
    --control.num_episodes=2 \
    --control.single_task="grasp."
```
"""

import logging
import socket
import time
import json
import numpy as np
import torch
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    stop_recording,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect, busy_wait
from lerobot.common.utils.utils import init_logging, log_say
from lerobot.configs import parser


@ControlConfig.register_subclass("server_record")
@dataclass
class ServerControlConfig(RecordControlConfig):
    """Configuration for server control mode."""
    listen_port: int = 9999  # Port to listen for client connections


@safe_disconnect
def server_record(
    robot: FairinoRobot,
    cfg: ServerControlConfig,
) -> LeRobotDataset:
    """Server record mode that receives leader arm data from client and controls follower arms."""
    if not robot.is_connected:
        robot.connect()
    
    # Initialize socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', cfg.listen_port))
    server_socket.listen(1)
    
    logging.info(f"Server listening on port {cfg.listen_port}")
    
    try:
        # Wait for client connection
        logging.info("Waiting for client connection...")
        client_socket, client_address = server_socket.accept()
        logging.info(f"Client connected from {client_address}")
        
        # Receive initial configuration from client
        init_config_data = client_socket.recv(1024).decode('utf-8')
        init_config = json.loads(init_config_data)
        
        # Acknowledge receipt
        client_socket.sendall("ACK".encode('utf-8'))
        
        # Create dataset for recording
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
        
        # Main recording loop
        episode_count = 0
        listener, events = init_keyboard_listener()
        
        # Process episodes until we reach the target number or stop recording
        while episode_count < cfg.num_episodes and not events["stop_recording"]:
            # Wait for episode start marker from client
            start_signal = client_socket.recv(1024).decode('utf-8')
            if start_signal != "START_EPISODE":
                if start_signal == "CLOSE":
                    logging.info("Client requested to close connection")
                    break
                logging.warning(f"Expected START_EPISODE, got: {start_signal}")
                continue
            
            logging.info(f"Starting to record episode {episode_count+1}")
            
            # Recording loop for current episode
            frame_count = 0
            start_episode_time = time.perf_counter()
            
            # Clear episode buffer in case we have data from a previous failed recording
            dataset.clear_episode_buffer()
            
            while not events["stop_recording"] and not events["exit_early"]:
                start_frame_time = time.perf_counter()
                
                # Receive data from client
                try:
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        logging.warning("Client disconnected")
                        break
                    
                    if data == "END_EPISODE":
                        logging.info("Episode ended by client")
                        break
                    
                    # Parse the received data
                    frame_data = json.loads(data)
                    
                    # Extract leader positions from received data and convert to tensors
                    leader_pos = {}
                    for arm_name, pos_list in frame_data["leader_pos"].items():
                        leader_pos[arm_name] = torch.tensor(pos_list, dtype=torch.float32)
                    
                    # Create action tensor from leader positions to control follower arms
                    action = []
                    for name in robot.follower_arms:
                        if name in leader_pos:
                            # Use corresponding leader arm position to control follower arm
                            # The positions have already been aligned in the client
                            action.append(leader_pos[name])
                    action = torch.cat(action)
                    
                    # Send action to robot
                    robot.send_action(action)
                    
                    # Capture current observation including camera images and follower arm positions
                    observation = robot.capture_observation()
                    
                    # Add leader arm positions to the observation for recording
                    # Create a new tensor that combines leader arm data for recording
                    leader_state = []
                    for name in leader_pos:
                        leader_state.append(leader_pos[name])
                    if leader_state:
                        leader_state_tensor = torch.cat(leader_state)
                        # Add to observation dict with a custom key for leader arms
                        observation["observation.leader_state"] = leader_state_tensor
                    
                    # Prepare action dict
                    action_dict = {"action": action}
                    
                    # Add frame to dataset
                    dataset.add_frame(observation, action_dict)
                    
                    # Acknowledge receipt to client
                    client_socket.sendall("ACK".encode('utf-8'))
                    
                    # Control FPS
                    elapsed_time = time.perf_counter() - start_frame_time
                    busy_wait(1.0/cfg.fps - elapsed_time)
                    
                    # Log performance info
                    dt_s = time.perf_counter() - start_frame_time
                    log_control_info(robot, dt_s, fps=cfg.fps)
                    
                    frame_count += 1
                    
                    # Check if episode time exceeded
                    if cfg.episode_time_s is not None and time.perf_counter() - start_episode_time > cfg.episode_time_s:
                        logging.info(f"Episode time limit reached ({cfg.episode_time_s}s)")
                        break
                
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON: {data}")
                    break
                except socket.error as e:
                    logging.error(f"Socket error: {e}")
                    break
            
            # Save the episode
            if frame_count > 0:
                dataset.save_episode(cfg.single_task)
                episode_count += 1
                logging.info(f"Saved episode {episode_count} with {frame_count} frames")
            
            # Signal to client that we're ready for the next episode
            client_socket.sendall("READY".encode('utf-8'))
            
            # Reset event flags
            events["exit_early"] = False
            
            # If user requested to stop recording
            if events["stop_recording"]:
                break
        
        # Stop recording and clean up
        logging.info("Stopping recording")
        stop_recording(robot, listener, cfg.display_cameras)
        
        # Consolidate dataset
        dataset.consolidate(cfg.run_compute_stats)
        logging.info("Dataset consolidated")
        
        return dataset
    
    except socket.error as e:
        logging.error(f"Socket error: {e}")
        return None
    finally:
        # Clean up sockets
        server_socket.close()
        logging.info("Server socket closed")


@parser.wrap()
def control_robot_server(cfg: ControlPipelineConfig):
    """Main server control function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Initialize robot with only follower arms and cameras
    robot = FairinoRobot(teleop_mode=False)
    
    # Run server record mode
    if isinstance(cfg.control, ServerControlConfig):
        server_record(robot, cfg.control)
    else:
        raise ValueError(f"Unsupported control type: {type(cfg.control)}")
    
    # Safe disconnect
    if robot.is_connected:
        robot.disconnect()


if __name__ == "__main__":
    # Register our custom control config
    control_robot_server()
