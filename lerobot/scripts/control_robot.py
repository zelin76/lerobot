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
from dataclasses import asdict
from pprint import pformat

# 从safetensors.torch导入load_file, save_file（当前已注释）
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
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
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# 控制模式实现
########################################################################################



@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    """遥操作模式，持续控制机器人运动"""
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,  # 启用遥操作模式
        display_cameras=False,  # 是否显示摄像头画面
    )


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


@parser.wrap()  # 参数解析装饰器
def control_robot(cfg: ControlPipelineConfig):
    """主控制函数，根据配置选择控制模式"""
    init_logging()
    logging.info(pformat(asdict(cfg)))  # 记录配置信息

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
    control_robot()  # 程序入口