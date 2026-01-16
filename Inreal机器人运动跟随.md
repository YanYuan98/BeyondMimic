# 仓库描述
本代码是基于BeyondMimic方法,基于GMR重定向的运动数据实现Inreal机器人运动跟踪
BeyondMimic仓库连接: [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

## 安装
- 安装 Isaac Lab v2.1.0 教程: [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- 下载仓库并安装
```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz

# 安装依赖项
python -m pip install -e source/whole_body_tracking
```

## 训练过程
### 数据预处理
- 基于[GMR](https://gitee.com/xmech/inrealgmrretarget)方法将人类数据重定向到Inreal机器人上，得到.csv数据文件
- 首先将csv的数据格式转换为npz的数据格式,这将自动将处理后的运动文件上传到 WandB 注册表，输出文件名为 {motion_name}。如果要添加新的机器人,则需要:
  - 在`source/whole_body_tracking/whole_body_tracking/assets/`中添加机器人模型,urdf文件
  - 在`source/whole_body_tracking/whole_body_tracking/robots/`中创建一个新的机器人配置文件,如`inreal.py`,修改其中参数
  - 修改`csv_to_npz_inreal.py`中的joint_names顺序,与motion_data中的关节顺序保持一致(如果GMR用的是xml文件,则其顺序就是xml文件中关节出现的顺序),建议都用urdf文件避免需要调整顺序的情况.

```bash
python scripts/csv_to_npz_inreal.py --robot_name inreal --input_file motion_data/walk_csv/run_inreal.csv --input_fps 30 --output_name run_inreal --headless
```
- 在 Isaac Sim 中重现运动来测试 WandB 注册表是否正常工作：
```bash
python scripts/replay_npz.py --registry_name=yanyuan98-mirrorme-org/wandb-registry-motions/run_inreal --robot_name inreal
```
  - Debugging
      - 请确保将 WANDB_EN​​TITY 环境变量设置为您的organization name，而不是您的 personal username.
      - 如果无法访问 /tmp 文件夹，请修改 csv_to_npz.py 文件中的第 319 行和第 326 行，将其指向您选择的临时文件夹。

### 训练

- 训练一个控制器

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train.py --task=Tracking-Flat-Inreal-Wo-State-Estimation-v2 --registry_name yanyuan98-mirrorme-org/wandb-registry-motions/36_09_walk_forward_turn_back --logger wandb --log_project_name BeyondMimic_Inreal_v2 --run_name walk_forward_turn_back --headless
```
`CUDA_VISIBLE_DEVICES=1`表示用第二个显卡训练，程序会自动将其映射为cuda:0
- 注意`tracking_env_cfg.py`的self.episode_length_s = 10.0时间与运动轨迹的时间长度

#### 训练Tips：
1. 通过GMR重映射后的动作，其运动轨迹光滑性显著影响控制器表现效果和sim-to-sim效果，因此可通过[robot-motion-editor](https://github.com/project-instinct/robot-motion-editor)编辑器对机器人轨迹进行光滑处理
2. beyondmimic自带的基于固有频率的pd参数计算方法会导致kp和kd项过大，不利于sim-to-sim和实物部署，建议重新设置

### 测试
#### 控制器测试
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-Inreal-Wo-State-Estimation-v2 --num_envs=1 --wandb_path=yanyuan98-zhejiang-university/BeyondMimic_Inreal/79hrhdbn --onnx_flag True --onnx_file ./logs/rsl_rl/inreal_v2_flat/2026-01-15_15-57-28_82_08_stageii_mod_final_v_2_1/2026-01-15_15-57-28_82_08_stageii_mod_final_v_2_1.onnx
```
- WandB run path 在run overview中。它遵循格式 {your_organization}/{project_name}/ 以及
具有唯一的 8 字符标识符。请注意，run_name 与 run_path 不同。
- `--onnx_flag`和`--onnx_file`用于测试onnx格式的控制器

#### sim to sim
在mujoco中测试机器人控制器onnx
```bash
python scripts/sim_to_sim/sim_to_sim.py --task Inreal_v2 --policy /home/yyy/Documents/Work/Imitation/BeyondMimic/logs/rsl_rl/inreal_v2_flat/2026-01-08_18-18-52_walk_forward_turn_back/2026-01-08_18-18-52_walk_forward_turn_back.onnx
```

## 主要代码

以下是此代码库的与task训练相关代码结构概览:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  此目录包含用于定义 BeyondMimic 的 MDP 的原子函数。以下是函数的详细说明:

    - **`commands.py`**
      用于根据参考运动、当前机器人状态和误差计算相关变量的命令库这包括位姿和速度误差计算、初始状态随机化和自适应采样

    - **`rewards.py`**
      实现了 DeepMimic 奖励函数和平滑项

    - **`events.py`**
      实现域随机化项

    - **`observations.py`**
      实现运动跟踪和数据采集的观测项

    - **`terminations.py`**
      实现提前终止和超时.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  包含跟踪任务的环境（MDP）超参数配置.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  包含跟踪任务的 PPO 超参数.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  包含机器人特定设置，包括骨架参数、关节刚度/阻尼计算和动作比例计算.

- **`scripts`**
  包含用于预处理运动数据、训练策略和评估已训练策略的实用脚本。


## 代码架构
### 训练代码
`scripts/rsl_rl/train.py`中定义训练代码
- task注册：`whole_body_tracking/tasks/Inreal_v2/__init__.py`：通过gym.register中注册了不同task对应的RL env、env_cfg、agent_cfg
- 任务选择：`train.py`中通过`@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")`动态选择不同的训练任务及其对应的cfg文件
- 环境创建：`env = gym.make(args_cli.task ...)`， make中基于--task的id通过`entry_point`，调用吗`isaaclab.envs:ManagerBasedRLEnv`创建训练环境
- 算法定义：`runner = OnPolicyRunner(env, agent_cfg.to_dict() ...)`基于rsl_rl库定义rl训练算法

### 运动控制代码
`whole_body_tracking/tasks/`中包含任务任务相关程序
- `./config/`中包含机器人训练用到的所有配置文件
  - 环境参数（缩进表示继承关系）: 
    - `flat_env_cfg.py`: 定义多个env_cfg类接口，如不同控制频率，状态变量是否包含身体位置和速度（可硬件部署）等
      - `Inreal_v2.py`：机器人参数文件，如pd参数，action scale系数，关节力矩、速度限制、转动惯量等
      - `tracking_env_cfg.py`：训练相关参数，如对环境基类scene，仿真基类sim设置参数；定义observations，actions，rewards等函数和参数
        - `ManagerBasedRLEnvCfg`：ManagerBased的RL相关的cfg基类定义，如observations，curriculum，rewards等的类型注解
          - `ManagerBasedEnvCfg`：ManagerBased的Env相关的cfg基类定义，如viewer，sim，scene等的类型注解
  - 算法参数（缩进表示继承关系）: 
    - `rsl_rl_ppo_cfg.py`: 网络大小和算法参数定义
      - `RslRlOnPolicyRunnerCfg`：算法参数基；类

`tracking_env_cfg.py`对象详解：
- `scene`：场景，`InteractiveSceneCfg`类型，并在`manager_based_env.py/__init__`通过`InteractiveScene(self.cfg.scene)`基于此cfg实例化scene
  - `InteractiveScene`：设置地形terrain、关节机器人articulations，可变形体deformable、刚体rigid、传感器sensors等场景元素
    - `Articulation`：在`InteractiveScene`根据`tracking_env_cfg.py`中的robot类型实例化，用以获取与机器人actuator（actuator_pd.idealpdactuator等）、joint_data、root、body等相关的所有数据，以及通过`wrie_*`相关函数从sim中获取最新状态，或者将最新数据写入sim。
- `command`：指令，`MotionCommandCfg`类型，并在`manager_based_env.py/__init__`实例化，在`mdp/command.py`中定义motion data的指令数据类型、command更新和自适应采样策略。
- `action`：动作，`ActionsCfg`类型，设置action类型以及相应的scale和offset等参数，`JointPositionAction`表示action代表关节目标位置，可通过`process_actions`和`apply_actions`等函数从网络输出action计算关节目标位置，并将其写入`Articulation`中
- `Observations`：观测量，`ObservationsCfg`类型，在`observation.py`中定义观测量获取函数，并在`managers/observation_manager.py`中在调用时compute所有观测量
- `event`：域随机化，`EventCfg`类型，在`events.py`中定义域随机化函数，并在`managers/event_manager.py`中在调用时compute所有量
- `reward`：奖励函数，`RewardsCfg`类型，在`reward.py`中定义奖励函数计算，并在`managers/reward_manager.py`中在调用时compute所有reward
- `terminations`：终止条件，`TerminationsCfg`类型，在`terminations.py`中定义终止条件，并在`managers/termination_manager.py`中在调用时compute所有终止条件


### 训练环境
训练环境在isaaclab库中的`source/isaaclab/isaaclab/`中定义
- Env：`envs/manager_based_rl_env.py`
  - `__init__`：scene创建；加载cfg文件；调用`managers/`中函数定义observations，actions，rewards，events等manager，定义状态和动作空间
    - 初始化时的环境reset：`env = RslRlVecEnvWrapper(env)`在符合rsl-rl的环境时会调用`env.reset`初始化所有所有场景、机器人等的状态,`env.reset`->`env._reset_idx`->`info = self.command_manager.reset(env_ids)`通过resample重置机器人状态
  - `step`：
    - action映射：通过self.`action_manager.process_action(action)`调用`ActionsCfg`定义的action的joint_action类型(如`JointPositionAction`)，基于该类型的`process_action`函数将action映射到target_pos
    - 设置关节目标位置或者力矩：调用`self.action_manager.apply_action`->`joint_action.apply_action`->`articulation.set_joint_position_target`函数设置各关节的目标位置或者力矩
    - 写入sim仿真：`self.scene.write_data_to_sim`->`articulation.write_data_to_sim`将上一步设置的目标位置写入仿真data中用内置pd控制器控制，或者通过idealpdactuator显示计算力矩
    - simulate：`self.sim.step`
    - 计算termination环境
    - 计算reward
    - 计算observation
    - reset终止的环境
  - `_reset_idx`：父类的reset函数调用`reset_idx`重置环境