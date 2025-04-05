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
python lerobot/scripts/control_robot_server.py --control.type=server_record --control.listen_port=9999 --control.fps=30 --control.repo_id=fr3/test3 --control.num_episodes=2 --control.single_task="grasp."
```
"""
import os
import cv2
import logging
import socket
import time
import json
import numpy as np
import torch
import select
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
from lerobot.scripts.baisic import *
from multiprocessing import Value
totolImage=np.zeros((240, 3*320,3),dtype=np.uint8)



server_level=Value('i',NotStarted)

@ControlConfig.register_subclass("server_record")
@dataclass
class ServerControlConfig(ControlConfig):
    """Configuration for server control mode."""
    listen_port: int = 8989  # Port to listen for client connections
    display_cameras : bool = True
    

@safe_disconnect
def server_record(
    robot: FairinoRobot,
    cfg: ServerControlConfig,
) -> LeRobotDataset:
    global totolImage, server_level
    
    """Server record mode that receives leader arm data from client and controls follower arms."""
    if not robot.is_connected:
        robot.connect()
    
    # Initialize socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', cfg.listen_port))
    server_socket.listen(1)
    server_socket.setblocking(False)
    client_socket = None
    logging.info(f"Server listening on port {cfg.listen_port}")
    
    try:
        listener, events = init_keyboard_listener()
        # Wait for client connection
        logging.info("Waiting for client connection...")
        
        server_level.value=WaitCon
        
        # 使用 select 函数监听所有连接的 client_socket
        while client_socket is None :
            readable_sockets, _, _ = select.select([server_socket], [], [], 1)
            # 处理所有可读的 socket
            for sock in readable_sockets:
                # 如果是 server_socket 表示有新的连接
                if sock is server_socket:
                    client_socket, client_address = server_socket.accept()
                    logging.info(f"Client connected from {client_address}")
                    break
        
        server_level.value=ClientConnected
        
        # Receive initial configuration from client
        readable_sockets, _, _ = select.select([client_socket], [], [], 10)
        init_config_data = client_socket.recv(1024).decode('utf-8')
        init_config = json.loads(init_config_data)
        print("recv client init config:",init_config)
        clear_camera_buffer_count = 100
        joint_pos = robot.capture_observation()["observation.state"].numpy()
        while clear_camera_buffer_count > 0 or not np.allclose(joint_pos, robot.config.initial_pos, atol=2.0):
            time.sleep(0.033)
            action = torch.from_numpy(robot.config.initial_pos)
            robot.send_action(action)
            joint_pos = robot.capture_observation()["observation.state"].numpy()
            clear_camera_buffer_count = clear_camera_buffer_count - 1
            
        # Acknowledge receipt
        client_socket.sendall("ACK".encode('utf-8'))
        ### parse client config 
        client_cfg_fps = init_config["fps"]
        client_cfg_repo_id = init_config["repo_id"]
        client_cfg_num_episodes = init_config["num_episodes"]
        client_cfg_single_task = init_config["single_task"]
        # Create dataset for recording
        #sanity_check_dataset_name(client_cfg_repo_id, cfg.policy)
 
        # Main recording loop
        episode_count = 0
        heritage=False
        if not os.path.isdir(os.path.join(dataset_dir, client_cfg_repo_id)) :
            dataset = LeRobotDataset.create(
                client_cfg_repo_id,
                client_cfg_fps,
                root=None,
                robot=robot,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
            )
            
        else:
            dataset=LeRobotDataset(client_cfg_repo_id)
            #recording continue
            episode_count=dataset.num_episodes
            client_cfg_num_episodes+=dataset.num_episodes
            heritage=True

        # Process episodes until we reach the target number or stop recording
        while episode_count < client_cfg_num_episodes :
            # Wait for episode start marker from client
            # 使用 select 函数监听所有连接的 client_socket
            _, _, _ = select.select([client_socket], [], [], 5)
            start_signal = client_socket.recv(1024).decode('utf-8')
            if not start_signal:
                logging.warning("Client disconnected")
                server_level.value=ConnectionClosed
                break
            if not start_signal.startswith("START_EPISODE"):
                if start_signal == "CLOSE":
                    logging.info("Client requested to close connection")
                    server_level.value=ConnectionClosed
                    break
                logging.warning(f"Expected START_EPISODE, got: {start_signal}")
                continue

            client_socket.sendall("ACK".encode('utf-8'))

            logging.info(f"Starting to record episode {episode_count+1}")
            
            # Recording loop for current episode
            frame_count = 0
            start_episode_time = time.perf_counter()
            
            # Clear episode buffer in case we have data from a previous failed recording
            if not heritage:
                dataset.clear_episode_buffer()
                heritage=False
                
            while True:
                start_frame_time = time.perf_counter()
                
                # Receive data from client
                try:
                    _, _, _ = select.select([client_socket], [], [],5)
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        logging.warning("Client disconnected")
                        break
                    if data == "END_EPISODE":
                        logging.info("Episode ended by client")
                        break
                    
                    # Acknowledge receipt to client
                    client_socket.sendall("ACK".encode('utf-8'))
                    # Parse the received data
                    frame_data = json.loads(data)
                    if "leader_pos" not in frame_data :
                        # Skip error frame data
                        continue
                    # Extract leader positions from received data and convert to tensors
                    leader_pos = {}
                    for arm_name, pos_list in frame_data["leader_pos"].items():
                        leader_pos[arm_name] = torch.tensor(pos_list, dtype=torch.float32)
                    #print("leader", leader_pos)
                    # Create action tensor from leader positions to control follower arms
                    action = []
                    for name in robot.follower_arms:
                        if name in leader_pos:
                            # Use corresponding leader arm position to control follower arm
                            # The positions have already been aligned in the client
                            action.append(leader_pos[name])
                    action.append(leader_pos["head"])
                    action = torch.cat(action)
                   
                    # Send action to robot
                    robot.send_action(action)
                    
                    # Capture current observation including camera images and follower arm positions
                    observation = robot.capture_observation()
                    
                    image_keys = [key for key in observation if "image" in key]
                    x_offset= 0
                    for key in image_keys:
                        
                        img = observation[key].numpy()                      
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        #putting all together
                        witdh=int(img.shape[1]/2)
                        totolImage[:,x_offset:x_offset+witdh]=img[:,0:witdh]
                        x_offset+=witdh
                    # Prepare action dict
                    cv2.imwrite("aux.jpg", totolImage)
                    
                    action_dict = {"action": action}
                    
                    # Add frame to dataset
                    frame = {**observation, **action_dict}
                    dataset.add_frame(frame)
                    
                    frame_count += 1
                    # Log performance info
                    dt_s = time.perf_counter() - start_frame_time
                    #print("server elapsed time:", dt_s*1000,"ms")
                    if frame_count % (client_cfg_fps * 2) == 0 :
                        log_control_info(robot, dt_s, fps=client_cfg_fps)
             
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON: {data}")
                    break
                except socket.error as e:
                    logging.error(f"Socket error: {e}")
                    break
            
            # Move robot to initial pos and record to dataset
            print("go to initial pos....")
            while True :
                start_frame_time = time.perf_counter()
                 # Create action tensor from robot initial pos
                action = torch.from_numpy(robot.config.initial_pos)
                # Send action to robot
                action_sent = robot.send_action(action)
                # Capture current observation including camera images and follower arm positions
                observation = robot.capture_observation()
                joint_pos = observation["observation.state"].numpy()
                if np.allclose(joint_pos, robot.config.initial_pos, atol=2.0):
                    break
                image_keys = [key for key in observation if "image" in key]
                x_offset= 0
                for key in image_keys:
                    img = observation[key].numpy()                      
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    #putting all together
                    witdh=int(img.shape[1]/2)
                    totolImage[:,x_offset:x_offset+witdh]=img[:,0:witdh]
                    x_offset+=witdh
                # Prepare action dict
                action_dict = {"action": action_sent}
                
                # Add frame to dataset
                frame = {**observation, **action_dict}
                dataset.add_frame(frame)
                
                frame_count += 1
                # Log performance info
                dt_s = time.perf_counter() - start_frame_time
                busy_wait(1.0/client_cfg_fps - dt_s)

            # Save the episode
            if frame_count > 0:
                dataset.save_episode(client_cfg_single_task)
                episode_count += 1
                logging.info(f"Saved episode {episode_count} with {frame_count} frames")
            
            # Signal to client that we're ready for the next episode
            client_socket.sendall("READY".encode('utf-8'))
        
        #reset server_level
        server_level.value=NotStarted
            
        # Stop recording and clean up
        logging.info("Stopping recording")
        stop_recording(robot, listener, cfg.display_cameras)
        
        # Consolidate dataset
        dataset.consolidate()
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
