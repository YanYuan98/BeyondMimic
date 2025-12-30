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
CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train.py --task=Tracking-Flat-Inreal-v0 --registry_name yanyuan98-mirrorme-org/wandb-registry-motions/run_inreal --logger wandb --log_project_name BeyondMimic_Inreal --run_name run_test --headless
```
`CUDA_VISIBLE_DEVICES=1`表示用第二个显卡训练，程序会自动将其映射为cuda:0

### 测试

- 控制器测试

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-Inreal-v0 --num_envs=1 --wandb_path=yanyuan98-zhejiang-university/BeyondMimic_Inreal/79hrhdbn (wandb-run-path)
```

WandB run path 在run overview中。它遵循格式 {your_organization}/{project_name}/ 以及
具有唯一的 8 字符标识符。请注意，run_name 与 run_path 不同。

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
