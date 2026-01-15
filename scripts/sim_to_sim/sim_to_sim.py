"""Script to test Isaac Lab trained policy in MuJoCo simulator (sim-to-sim transfer)."""

import argparse
import os
import yaml
import time
import numpy as np
import mujoco
import mujoco.viewer
from MujocoRobotEnv import MuJoCoRobotEnv

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
    
    while viewer.is_running() and not should_exit:        
        obs = env.reset()
        done = False
        step_count = 0
        action = np.zeros(env.model.nu)

        # break
        
        while not done and step_count < args.max_steps and viewer.is_running() and not should_exit:
            time.sleep(0.01)  # 控制频率
            
            # 推理动作
            obs_tensor = obs.reshape(1, -1).astype(np.float32)

            if env.time_steps == 1:
                print("="*60)
                print("time_steps: ", env.time_steps)
                print("obs motion pos: ", obs[0:20])
                print("obs motion vel: ", obs[20:40])
                print("obs_quat: ", obs[40:46])
                print("obs_ang_vel: ", obs[46:49])
                print("obs_joint_pos: ", obs[49:69])
                print("obs_joint_vel: ", obs[69:89])
                print("last_action: ", obs[89:109])
                print("armature: ", env.model.dof_armature)

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


if __name__ == "__main__":
    main()