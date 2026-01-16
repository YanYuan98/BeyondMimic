"""Script to test Isaac Lab trained policy in MuJoCo simulator (sim-to-sim transfer)."""

import argparse
import os
import yaml
import time
import numpy as np
import mujoco
import mujoco.viewer
from MujocoRobotEnv import MuJoCoRobotEnv
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    parser = argparse.ArgumentParser(description="Test Isaac Lab policy in MuJoCo")
    parser.add_argument("--task", type=str, required=True, 
                       help="robot type")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to ONNX policy file")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum steps per episode")
    parser.add_argument("--save_video", action="store_true",
                       help="Save video of the simulation")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.policy):
        raise FileNotFoundError(f"Policy file not found: {args.policy}")
    
    if args.task == "Inreal_v2":
        model_file = "/home/yyy/Documents/Work/Imitation/BeyondMimic/source/" \
        "whole_body_tracking/whole_body_tracking/assets/inreal_v2_description/xml/inreal_v2.xml"
        cfg_file = "/home/yyy/Documents/Work/Imitation/BeyondMimic/scripts/sim_to_sim/default_cfg.yaml"
    
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    body_names = [
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
    ]

    anchor_body_name = "torso_link"
    
    # 创建环境
    env = MuJoCoRobotEnv(
        model_path=model_file,
        policy_path=args.policy,
        cfg = cfg,
        body_names=body_names,
        anchor_body_name=anchor_body_name,
        init_noise = True
    )

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    should_exit = False
    visual_falg = True
    
    while viewer.is_running() and not should_exit:        
        obs = env.reset()
        done = False
        step_count = 0
        action = np.zeros(env.model.nu)

        joint_pos = np.zeros(env.model.nu)
        joint_pos_ref = np.zeros(env.model.nu)
        joint_vel = np.zeros(env.model.nu)
        torque = np.zeros(env.model.nu)
        tt = []
        # break
        
        while not done and step_count < args.max_steps and viewer.is_running() and not should_exit:
            time.sleep(0.01)  # 控制频率

            joint_pos = np.vstack((joint_pos, env.data.qpos[7:]))
            joint_vel = np.vstack((joint_vel, env.data.qvel[6:]))
            torque = np.vstack((torque, env.data.ctrl[:]))
            joint_pos_ref = np.vstack((joint_pos_ref, obs[0:20]))
            tt.append(env.data.time)
            
            # 推理动作
            obs_tensor = obs.reshape(1, -1).astype(np.float32)

            if env.time_steps <= 3:
                print("="*60)
                print("time_steps: ", env.time_steps)
                print("obs motion pos: ", obs[0:20])
                print("robot joint pos: ", env.data.qpos[7:])
                print("obs motion vel: ", obs[20:40])
                print("obs_quat: ", obs[40:46])
                print("obs_ang_vel: ", obs[46:49])
                print("obs_joint_pos: ", obs[49:69])
                print("obs_joint_vel: ", obs[69:89])
                print("last_action: ", obs[89:109])

            action, _ = env.get_session_output(obs_tensor)
            action = action.flatten()

            if env.time_steps == 1:
                print("action: ", action)
                print("="*60)
            
            # 执行动作
            obs, _, done, _ = env.step(action)
            
            # 渲染
            viewer.sync()
            
            step_count += 1        
        if done:
            print("Episode terminated early (robot fell)")

    if visual_falg:
        params = {
            'text.usetex': True,
            'font.size': 6,
            'font.family': 'Times New Roman',
            # 'image.cmap': 'summer',
            'image.cmap': 'GnBu',
            'axes.titlesize': 8,
            'legend.fontsize': 6,
            'axes.labelsize': 6,
            'lines.linewidth': 1.0,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'axes.titlepad': 2.0,
            'axes.labelpad': 2.0,
            'xtick.major.pad': 0.5,           # x轴刻度标签距离坐标轴的距离
            'ytick.major.pad': 0.5,           # y轴刻度标签距离坐标轴的距离
            'lines.markersize': 2,
            'figure.subplot.wspace': 0.5,
            'figure.subplot.hspace': 0.5,}
        mpl.rcParams.update(params)

        fig1, ax1 = plt.subplots(4, 6, figsize=(12.0, 8.0), dpi=300)
        fig2, ax2 = plt.subplots(4, 6, figsize=(12.0, 8.0), dpi=300)
        fig3, ax3 = plt.subplots(4, 6, figsize=(12.0, 8.0), dpi=300)
        for i in range(env.model.nu):
            ax1[i//6][i%6].plot(tt, joint_pos[1:, i], label="q")
            ax1[i//6][i%6].plot(tt, joint_pos_ref[1:, i], label="q_ref")
            ax2[i//6][i%6].plot(tt, joint_vel[1:, i])
            ax3[i//6][i%6].plot(tt, torque[1:, i])

        fig1.suptitle("Joint Pos")
        fig2.suptitle("Joint Vel")
        fig3.suptitle("Torque")
        plt.show()

if __name__ == "__main__":
    main()