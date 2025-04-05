"""Server utility for the FR3 robot control system."""
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
from multiprocessing import Manager,Queue,shared_memory

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

# Create shared memory manager
manager = Manager()
shared_dict = manager.dict()
#shared_dict['totolImage'] = manager.Array('B', 240*3*320*3)  # 240h, 960w (3*320), 3 channels
shared_dict['server_level'] = manager.Value('i', NotStarted)
#shm=shared_memory.SharedMemory(create=True, size=320*240*3*3)
#totoImage=np.ndarray([240,320,3],dtype=np.uint8,buffer=shm.buf)

totoImage=np.zeros((240, 960, 3), dtype=np.uint8)
queue=Queue()
def child_producer(shm, queue, image):
    """子进程：生成图像并写入共享内存"""
    # 模拟生成一张随机RGB图像（480x640x3）
    #image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # 创建共享内存并写入数据
    shm_array = np.ndarray(totoImage.shape, dtype=totoImage.dtype, buffer=shm.buf)
    np.copyto(shm_array, image)
    
    # 将共享内存名称和图像参数通过Queue传递给父进程
    queue.put({
        "name": shm.name,
        "shape": image.shape,
        "dtype": image.dtype
    })
    print("[子进程] 图像数据已写入共享内存，名称已发送")

def producer(image, shared_dict):
    # 读取图像并转换为字节流   
    _, img_encoded = cv2.imencode(".jpg", image)  # 压缩为字节流
    shared_dict["img_data"] = img_encoded.tobytes()  # 存储字节数据
    shared_dict["shape"] = image.shape  # 存储图像形状  



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
    global shared_dict,queue
    totolImage=np.zeros((240, 3*320,3),dtype=np.uint8) # aux only 
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
    shm = shared_memory.SharedMemory(create=True, size=totoImage.nbytes)
    
    try:
        listener, events = init_keyboard_listener()
        logging.info("Waiting for client connection...")
        
        shared_dict['server_level'].value = WaitCon
        
        while client_socket is None:
            readable_sockets, _, _ = select.select([server_socket], [], [], 1)
            for sock in readable_sockets:
                if sock is server_socket:
                    client_socket, client_address = server_socket.accept()
                    logging.info(f"Client connected from {client_address}")
                    break
        
        shared_dict['server_level'].value = ClientConnected
        
        # Receive initial configuration from client
        readable_sockets, _, _ = select.select([client_socket], [], [], 10)
        init_config_data = client_socket.recv(1024).decode('utf-8')
        init_config = json.loads(init_config_data)
        print("recv client init config:",init_config)
        clear_camera_buffer_count = 100
        while clear_camera_buffer_count:
            robot.capture_observation()
            clear_camera_buffer_count -= 1
            time.sleep(0.01)
            
        client_socket.sendall("ACK".encode('utf-8'))
        client_cfg_fps = init_config["fps"]
        client_cfg_repo_id = init_config["repo_id"]
        client_cfg_num_episodes = init_config["num_episodes"]
        client_cfg_single_task = init_config["single_task"]

        # Main recording loop
        episode_count = 0
        heritage = False
        if not os.path.isdir(os.path.join(dataset_dir, client_cfg_repo_id)): #recording for new dataset
            dataset = LeRobotDataset.create(
                client_cfg_repo_id,
                client_cfg_fps,
                root=None,
                robot=robot,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
            )
        else: #recording from existing dataset
            dataset = LeRobotDataset(client_cfg_repo_id)
            episode_count = dataset.num_episodes
            client_cfg_num_episodes += dataset.num_episodes
            heritage = True

        while episode_count < client_cfg_num_episodes:
            _, _, _ = select.select([client_socket], [], [], 5)
            start_signal = client_socket.recv(1024).decode('utf-8')
            if not start_signal:
                logging.warning("Client disconnected")
                shared_dict['server_level'].value = ConnectionClosed
                break

            if not start_signal.startswith("START_EPISODE"):
                if start_signal == "CLOSE":
                    logging.info("Client requested to close connection")
                    shared_dict['server_level'].value = ConnectionClosed
                    break
                logging.warning(f"Expected START_EPISODE, got: {start_signal}")
                continue

            client_socket.sendall("ACK".encode('utf-8'))
            logging.info(f"Starting to record episode {episode_count+1}")
            
            frame_count = 0
            start_episode_time = time.perf_counter()
            
            if not heritage:
                dataset.clear_episode_buffer()
                heritage = False
                
            while True:
                start_frame_time = time.perf_counter()
                
                try:
                    _, _, _ = select.select([client_socket], [], [],5)
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        logging.warning("Client disconnected")
                        break
                    if data == "END_EPISODE":
                        logging.info("Episode ended by client")
                        break
                    
                    client_socket.sendall("ACK".encode('utf-8'))
                    frame_data = json.loads(data)
                    
                    leader_pos = {}
                    for arm_name, pos_list in frame_data["leader_pos"].items():
                        leader_pos[arm_name] = torch.tensor(pos_list, dtype=torch.float32)
                        
                    action = []
                    for name in robot.follower_arms:
                        if name in leader_pos:
                            action.append(leader_pos[name])
                    action.append(leader_pos["head"])
                    action = torch.cat(action)
                   
                    robot.send_action(action)
                    observation = robot.capture_observation()
                    
                    # Update shared memory image
                    x_offset = 0
                    image_keys = [key for key in observation if "image" in key]
                    
                    for key in image_keys:
                        img = observation[key].numpy()
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        width = int(img.shape[1]/2)
                        
                        totolImage[:,x_offset:x_offset+width] = img[:,0:width]
                        x_offset += width
                    child_producer(shm=shm, image=totolImage,queue=queue)
                    #producer(image=totolImage,shared_dict=shared_dict)
                    action_dict = {"action": action}
                    frame = {**observation, **action_dict}
                    dataset.add_frame(frame)
                    frame_count += 1

                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON: {data}")
                    break
                except socket.error as e:
                    logging.error(f"Socket error: {e}")
                    break

            if frame_count > 0:
                dataset.save_episode(client_cfg_single_task)
                episode_count += 1
                logging.info(f"Saved episode {episode_count} with {frame_count} frames")
            
            client_socket.sendall("READY".encode('utf-8'))

        shared_dict['server_level'].value = NotStarted
        logging.info("Stopping recording")
        stop_recording(robot, listener, cfg.display_cameras)
        dataset.consolidate()
        logging.info("Dataset consolidated")
        return dataset
    
    except socket.error as e:
        logging.error(f"Socket error: {e}")
        return None
    finally:
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
