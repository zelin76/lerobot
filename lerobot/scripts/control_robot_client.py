"""
Client utility for the FR3 robot control system.

This script connects only to the leader_arms of the FR3 robot and sends
the joint positions data to the server via socket connection.

Example usage:
```bash
python lerobot/scripts/control_robot_client.py     --control.type=client_record     --control.listen_ip=10.0.0.24    --control.listen_port=9999   --control.fps=30     --control.repo_id=fr3/test3     --control.num_episodes=2     --control.single_task="grasp."
```
"""

import logging
import socket
import select
import time
import json
import numpy as np
import torch
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    busy_wait,
)
from lerobot.common.robot_devices.robots.fr3_robot import FairinoRobot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser


@ControlConfig.register_subclass("client_record")
@dataclass
class ClientControlConfig(RecordControlConfig):
    """Configuration for client control mode."""
    listen_ip: str = "localhost"  # IP of the server to connect to
    listen_port: int = 8989      # Port of the server to connect to


@safe_disconnect
def client_record(
    robot: FairinoRobot,
    cfg: ClientControlConfig,
):
    """Client record mode that sends leader arm data to the server."""
    if not robot.is_connected:
        robot.connect()

    # Initialize socket connection to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    logging.info(f"Connecting to server at {cfg.listen_ip}:{cfg.listen_port}")
    
    try:
        listener, events = init_keyboard_listener()
        client_socket.connect((cfg.listen_ip, cfg.listen_port))
        client_socket.setblocking(False)
        logging.info("Connected to server successfully")
        
        # Initialize keyboard listener for control flow
        
        
        # Send initial configuration to server
        init_config = {
            "fps": cfg.fps,
            "repo_id": cfg.repo_id,
            "num_episodes": cfg.num_episodes,
            "single_task": cfg.single_task
        }
        client_socket.sendall(json.dumps(init_config).encode('utf-8'))
        
        # Wait for server acknowledgment
        readable_sockets, _, _ = select.select([client_socket], [], [], 10)
        response = client_socket.recv(1024).decode('utf-8')
        if response != "ACK":
            logging.error(f"Server did not acknowledge: {response}")
            return
        
        logging.info("Starting to send leader arm data")
        
        # Main control loop
        episode_count = 0
        while episode_count < cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"Starting episode {episode_count+1}")
            
            # Send episode start marker
            client_socket.sendall("START_EPISODE".encode('utf-8'))
            
            # Recording loop for current episode
            frame_count = 0
            start_episode_time = time.perf_counter()
            
            while not events["stop_recording"] and not events["exit_early"]:
                start_frame_time = time.perf_counter()
                
                # Read leader arm positions
                leader_pos = robot.get_leader_pos()
 
                # Convert to JSON and send
                data_to_send = {
                    "frame_count": frame_count,
                    "leader_pos": {k: v.tolist() for k, v in leader_pos.items()}
                }
                client_socket.sendall(json.dumps(data_to_send).encode('utf-8'))
                #print("sent:",data_to_send)
                # Receive acknowledgment from server
                readable_sockets, _, _ = select.select([client_socket], [], [], 1000)
                ack = client_socket.recv(1024).decode('utf-8')
                if ack != "ACK":
                    logging.warning(f"Unexpected server response: {ack}")
                # Control FPS
                elapsed_time = time.perf_counter() - start_frame_time
                busy_wait(1.0/cfg.fps - elapsed_time)
                print("client elapsed time:", elapsed_time*1000,"ms")
                frame_count += 1

                dt_s = time.perf_counter() - start_frame_time
                if frame_count % (cfg.fps * 2) == 0 :
                    log_control_info(robot, dt_s, fps=cfg.fps)
                
                # Check if episode time exceeded
                if cfg.episode_time_s is not None and time.perf_counter() - start_episode_time > cfg.episode_time_s:
                    logging.info(f"Episode time limit reached ({cfg.episode_time_s}s)")
                    break
            
            # Send episode end marker
            client_socket.sendall("END_EPISODE".encode('utf-8'))
            print("Waiting for server save data ready ....")
            # Wait for server to be ready for next episode
            while 1:
                readable_sockets, _, _ = select.select([client_socket], [], [], 100)
                ready = client_socket.recv(1024).decode('utf-8')
                if ready.find('READY') != -1 :
                    logging.error(f"Server ready for next episode: {ready}")
                    break
                else :
                    print("Waiting for server save data ready ....")
               
            # Reset event flags
            events["exit_early"] = False
            
            # If user requested to stop recording
            if events["stop_recording"]:
                break
            else :
                while not events["exit_early"]:
                    print("Waitting for env recover ...")
                    print("Press > to continue")
                    time.sleep(1)
                # Reset event flags
                events["exit_early"] = False
            episode_count += 1

        # Close connection properly
        client_socket.sendall("CLOSE".encode('utf-8'))
        
        if listener:
            listener.stop()
        
        logging.info("Client recording completed")
        
    except ConnectionRefusedError:
        logging.error(f"Failed to connect to server at {cfg.listen_ip}:{cfg.listen_port}")
    except socket.error as e:
        logging.error(f"Socket error: {e}")
    finally:
        client_socket.close()
        logging.info("Socket connection closed")


@parser.wrap()
def control_robot_client(cfg: ControlPipelineConfig):
    """Main client control function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    left_com=cfg.control.left_com  
    right_com=cfg.control.right_com
    # Initialize robot with only leader arms (no follower arms or cameras)
    custom_robot = FairinoRobot(teleop_mode=True, left_com=left_com, right_com=right_com)
    
    # Run client record mode
    if isinstance(cfg.control, ClientControlConfig):
        client_record(custom_robot, cfg.control)
    else:
        raise ValueError(f"Unsupported control type: {type(cfg.control)}")
    
    # Safe disconnect
    if custom_robot.is_connected:
        custom_robot.disconnect()


if __name__ == "__main__":
    control_robot_client()
