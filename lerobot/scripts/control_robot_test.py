"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

用于控制机器人的实用工具。

支持校准机器人、遥操作、记录数据集、回放数据集等功能。
通过命令行参数配置不同的控制模式和机器人类型。

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.
If the dataset you want to extend is not on the hub, you also need to add `--control.local_files_only=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
import socket
import json
import threading
import argparse
import sys
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Dict, List, Optional, Union, Any
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)

# 定义新的客户端和服务器控制配置
@dataclass
class ClientControlConfig:
    """客户端控制配置，针对领导者手臂"""
    server_ip: str = "127.0.0.1"  # 服务器IP地址
    server_port: int = 8888       # 服务器端口
    fps: int = 30                 # 控制频率
    teleop_time_s: float = float("inf")  # 控制时间，默认无限
    display_cameras: bool = True  # 是否显示摄像头
    
@dataclass
class ServerControlConfig:
    """服务器控制配置，针对跟随者手臂"""
    listen_ip: str = "0.0.0.0"    # 监听地址，默认所有网络接口
    listen_port: int = 8888       # 监听端口
    fps: int = 30                 # 控制频率
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

########################################################################################
# 控制模式实现
########################################################################################



# 数据传输类，用于序列化和反序列化传输的动作数据
@dataclass
class ActionData:
    """机器人动作数据类，用于网络传输"""
    action: List[float] = field(default_factory=list)  # 动作数据
    timestamp: float = 0.0        # 时间戳
    
    def to_json(self) -> str:
        """将动作数据转换为JSON字符串"""
        return json.dumps({"action": self.action, "timestamp": self.timestamp})
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ActionData':
        """从JSON字符串创建动作数据对象"""
        data = json.loads(json_str)
        return cls(action=data["action"], timestamp=data["timestamp"])


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    """遥操作模式，持续控制机器人运动"""
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,  # 启用遥操作模式
        display_cameras=True,  # 是否显示摄像头画面
    )


@safe_disconnect
def run_client(robot: Robot, cfg: ClientControlConfig):
    """客户端模式（领导者手臂），将动作数据发送到服务器"""
    logging.info(f"启动客户端模式，连接到服务器 {cfg.server_ip}:{cfg.server_port}")
    
    # 创建Socket客户端
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((cfg.server_ip, cfg.server_port))
        logging.info("已连接到服务器")
        
        # 开始控制循环
        start_time = time.perf_counter()
        frame_count = 0
        
        while True:
            frame_start_time = time.perf_counter()
            
            # 获取当前机器人状态（此处假设robot.get_state()返回当前关节状态）
            current_state = robot.get_state() if hasattr(robot, "get_state") else robot.get_current_joint_positions()
            
            # 封装动作数据
            action_data = ActionData(
                action=current_state if isinstance(current_state, list) else current_state.tolist(),
                timestamp=time.time()
            )
            
            # 发送数据到服务器
            try:
                client_socket.sendall((action_data.to_json() + "\n").encode("utf-8"))
            except ConnectionError as e:
                logging.error(f"发送数据失败: {e}")
                break
                
            # 从服务器接收确认（可选）
            # response = client_socket.recv(1024).decode("utf-8").strip()
            # logging.debug(f"服务器响应: {response}")
            
            # 运行本地控制逻辑
            if hasattr(robot, "teleop_update"):
                robot.teleop_update()  # 更新遥操作状态
            
            # 显示摄像头（如果需要）
            if cfg.display_cameras and hasattr(robot, "render_cameras"):
                robot.render_cameras()
                
            # 帧率控制
            dt_s = time.perf_counter() - frame_start_time
            busy_wait(1 / cfg.fps - dt_s)
            
            frame_count += 1
            elapsed_time = time.perf_counter() - start_time
            
            # 显示控制信息
            if frame_count % 10 == 0:
                current_fps = frame_count / elapsed_time
                logging.info(f"已运行 {elapsed_time:.2f} 秒，当前 FPS: {current_fps:.1f}")
            
            # 检查是否达到控制时间限制
            if elapsed_time >= cfg.teleop_time_s:
                logging.info("达到控制时间限制，退出客户端")
                break
                
    except Exception as e:
        logging.error(f"客户端运行错误: {e}")
    finally:
        # 关闭Socket
        client_socket.close()
        logging.info("客户端已关闭")


@safe_disconnect
def run_server(robot: Robot, cfg: ServerControlConfig):
    """服务器模式（跟随者手臂），接收客户端发送的动作数据并执行"""
    logging.info(f"启动服务器模式，监听 {cfg.listen_ip}:{cfg.listen_port}")
    
    # 创建Socket服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((cfg.listen_ip, cfg.listen_port))
        server_socket.listen(1)  # 最多接受1个连接
        logging.info("等待客户端连接...")
        
        client_conn, client_addr = server_socket.accept()
        logging.info(f"客户端 {client_addr} 已连接")
        
        buffer = ""
        start_time = time.perf_counter()
        frame_count = 0
        
        while True:
            # 接收数据
            try:
                data = client_conn.recv(4096).decode("utf-8")
                if not data:
                    logging.info("客户端已断开连接")
                    break
                    
                buffer += data
                
                # 处理可能的多个或部分消息
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    
                    # 解析动作数据
                    try:
                        action_data = ActionData.from_json(line)
                        
                        # 将动作应用到机器人
                        robot.send_action(action_data.action)
                        
                        # 可选：发送确认消息
                        # client_conn.sendall("ACK\n".encode("utf-8"))
                        
                        frame_count += 1
                        if frame_count % 10 == 0:
                            elapsed = time.perf_counter() - start_time
                            logging.info(f"已接收 {frame_count} 帧，当前 FPS: {frame_count/elapsed:.1f}")
                            
                    except json.JSONDecodeError:
                        logging.error(f"无效的JSON数据: {line}")
                        continue
            except ConnectionError:
                logging.info("客户端连接已断开")
                break
                
    except Exception as e:
        logging.error(f"服务器运行错误: {e}")
    finally:
        # 关闭连接
        if 'client_conn' in locals():
            client_conn.close()
        server_socket.close()
        logging.info("服务器已关闭")


@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    """记录数据集模式，支持多episode录制和策略控制"""
    # 处理数据集恢复/继续录制
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            local_files_only=cfg.local_files_only,
        )
        # 初始化图像写入器（多进程/线程）
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        # 校验数据集与机器人兼容性
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # 创建新数据集
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

    # 加载预训练策略（如果有）
    policy = None if cfg.policy is None else make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()
    # 初始化键盘监听（用于控制录制流程）
    listener, events = init_keyboard_listener()
    # 录制前热身阶段（调整起始位置/设备同步）
    enable_teleoperation = policy is None
    log_say("回到初始位置", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)
    log_say("准备开始录制", cfg.play_sounds)
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()  # 安全停止检查

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        # 开始录制单个episode
        log_say(f"正在录制第{dataset.num_episodes}个episode", cfg.play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_cameras=cfg.display_cameras,
            policy=policy,
            device=cfg.device,
            use_amp=cfg.use_amp,
            fps=cfg.fps,
        )

        # 环境重置阶段
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("正在重置环境", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s)

        # 处理重新录制逻辑
        if events["rerecord_episode"]:
            log_say("重新录制当前episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()  # 清空当前episode缓存
            continue

        # 保存当前episode
        dataset.save_episode(cfg.single_task)
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    # 停止录制后续处理
    log_say("停止录制", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_cameras)

    # 计算数据集统计信息（可选）
    if cfg.run_compute_stats:
        logging.info("正在计算数据集统计信息")

    dataset.consolidate(cfg.run_compute_stats)

    log_say("退出程序", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    """回放模式，执行数据集中的动作序列"""
    # 加载指定episode的数据集
    dataset = LeRobotDataset(
        cfg.repo_id, root=cfg.root, episodes=[cfg.episode], local_files_only=cfg.local_files_only
    )
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("开始回放episode", cfg.play_sounds, blocking=True)
    # 逐帧执行动作
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)  # 发送动作指令

        # 精确控制帧率
        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)  # 记录控制信息


@dataclass
class DirectControlModeConfig:
    """直接控制模式配置（用于客户端/服务器模式）"""
    mode: str = ""  # 模式: "client" 或 "server"
    
    # 客户端特定参数
    server_ip: str = "127.0.0.1"  # 服务器IP
    server_port: int = 8888       # 服务器端口
    display_cameras: bool = True  # 显示摄像头
    
    # 服务器特定参数
    listen_ip: str = "0.0.0.0"    # 监听IP
    listen_port: int = 8888       # 监听端口
    
    # 通用参数
    fps: int = 30                 # 帧率
    teleop_time_s: float = float("inf")  # 控制时间


@parser.wrap()  # 参数解析装饰器
def control_robot(cfg: ControlPipelineConfig):
    """主控制函数，根据配置选择控制模式"""
    init_logging()
    logging.info(pformat(asdict(cfg)))  # 记录配置信息

    # 处理client/server直接模式：通过第一个位置参数
    direct_mode = None
    if len(sys.argv) > 1 and sys.argv[1] in ["client", "server"]:
        direct_mode = sys.argv[1]
        direct_cfg = DirectControlModeConfig(mode=direct_mode)
        
        # 解析其余的命令行参数（不使用argparse以保持与原参数解析的兼容性）
        for i in range(2, len(sys.argv)):
            arg = sys.argv[i]
            if not arg.startswith("--"):
                continue
                
            parts = arg.split("=", 1)
            key = parts[0][2:]  # 移除前缀'--'
            value = parts[1] if len(parts) > 1 else True
            
            # 客户端特定参数
            if key == "server-ip" and direct_mode == "client":
                direct_cfg.server_ip = value
            elif key == "server-port" and direct_mode == "client":
                direct_cfg.server_port = int(value)
            elif key == "display-cameras" and direct_mode == "client":
                direct_cfg.display_cameras = value.lower() == "true"
            # 服务器特定参数
            elif key == "listen-ip" and direct_mode == "server":
                direct_cfg.listen_ip = value
            elif key == "listen-port" and direct_mode == "server":
                direct_cfg.listen_port = int(value)
            # 通用参数
            elif key == "fps":
                direct_cfg.fps = int(value)
                
        # 根据模式创建机器人和执行相应功能
        if direct_mode == "client":
            client_cfg = ClientControlConfig(
                server_ip=direct_cfg.server_ip,
                server_port=direct_cfg.server_port,
                fps=direct_cfg.fps,
                display_cameras=direct_cfg.display_cameras
            )
            logging.info(f"启动客户端模式: {pformat(asdict(client_cfg))}")
            robot = FairinoRobot(teleop_mode=True)
            run_client(robot, client_cfg)
            
        elif direct_mode == "server":
            server_cfg = ServerControlConfig(
                listen_ip=direct_cfg.listen_ip,
                listen_port=direct_cfg.listen_port,
                fps=direct_cfg.fps
            )
            logging.info(f"启动服务器模式: {pformat(asdict(server_cfg))}")
            robot = FairinoRobot(teleop_mode=False)
            run_server(robot, server_cfg)
            
        # 安全断开连接
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect()
        return
    
    # 原有控制模式逻辑
    teleop_mode = not isinstance(cfg.control, ReplayControlConfig)
    robot = FairinoRobot(teleop_mode=teleop_mode)  # 根据配置创建机器人实例

    # 根据控制类型选择执行模式
    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)

    # 安全断开连接
    if robot.is_connected:
        robot.disconnect()


if __name__ == "__main__":
    # 所有模式都通过主入口函数处理，这样能确保兼容原有的参数处理逻辑
    control_robot()  # 程序入口
