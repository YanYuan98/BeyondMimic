"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--onnx_flag", type=bool, default=False, help="Use onnx model.")
parser.add_argument("--onnx_file", type=str, default=None, help="Path to the onnx file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import onnxruntime as ort


# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # disable randomization, noise, and disturbance forces for clean play
    # NOTE: do NOT set `env_cfg.events = None` because manager code expects an events
    # config object. Instead, disable individual event terms so the managers keep a valid cfg.
    if getattr(env_cfg, "events", None) is not None:
        for _name, _term in env_cfg.events.__dict__.items():
            if _term is None:
                continue
            # try to disable by clearing the execution mode if available
            if hasattr(_term, "mode"):
                try:
                    _term.mode = None
                except Exception:
                    # fall back to removing the term
                    try:
                        setattr(env_cfg.events, _name, None)
                    except Exception:
                        pass
    
    # disable observation noise
    for obs_group in [env_cfg.observations.policy, env_cfg.observations.critic]:
        for obs_term_name, obs_term in obs_group.__dict__.items():
            if obs_term is not None and hasattr(obs_term, "noise"):
                obs_term.noise = None
                print(f"[INFO] Disabled noise for observation term: {obs_term_name}")
    print("[INFO] Disabled randomization, noise, and disturbance forces for play mode.")

    # Enable test mode for motion command (always start from time_steps=0 without random sampling)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.is_test_mode = True
        print("[INFO] Enabled test mode for motion command.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    robot = env.unwrapped.scene["robot"]
    # Ensure initial joint positions and velocities are zero for clean controller tests
    try:
        with torch.no_grad():
            if hasattr(robot.data, "joint_pos"):
                robot.data.joint_pos[:] = torch.zeros_like(robot.data.joint_pos)
            if hasattr(robot.data, "joint_vel"):
                robot.data.joint_vel[:] = torch.zeros_like(robot.data.joint_vel)
    except Exception:
        pass

    joint_pos = robot.data.joint_pos
    body_pos_w = robot.data.body_pos_w
    print(f"Robot init joint_pos: {joint_pos[0].cpu().numpy()}")
    if hasattr(robot.data, "joint_vel"):
        print(f"Robot init joint_vel: {robot.data.joint_vel[0].cpu().numpy()}")
    
    timestep = 0

    root_pos = robot.data.body_pos_w[0, 0].cpu().numpy()    # (x, y, z)
    root_quat = robot.data.body_quat_w[0, 0].cpu().numpy()  # (w, x, y, z)

    print("root_pos:", root_pos)
    print("root_quat (w,x,y,z):", root_quat)
    
    if args_cli.onnx_flag == True:
        if args_cli.onnx_file is None:
            print("[ERROR] onnx_file argument is required when onnx_flag is True.")
            return
        print("[Onnx] Using Onnx model as policy model.")
        onnx_path = args_cli.onnx_file
        # onnx_path = "/home/yyy/Documents/Work/Imitation/BeyondMimic/logs/rsl_rl/" \
        # "inreal_v2_flat/2026-01-08_18-18-52_walk_forward_turn_back/" \
        # "2026-01-08_18-18-52_walk_forward_turn_back.onnx"
        session = ort.InferenceSession(onnx_path)
    else:
        print("[No Onnx] Not Using Onnx model as policy model.")

    # Print runtime joint name list and index mapping (helps map obs/actions -> joint names)
    try:
        print("Runtime joint order (index: name):")
        for i, name in enumerate(robot.joint_names):
            # print(f"{i}: {name}")
            print(f"joint_id = {i:02d}, joint_name = {name}")
    except Exception:
        pass

    obs, _ = env.get_observations()
    # 从底层环境中获取 MotionCommand 对象（使用 unwrapped 访问底层环境）
    motion_command = env.unwrapped.command_manager.get_term("motion")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get current robot joint states (位置和速度)
            try:
                robot = env.unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos  # shape: (num_envs, num_joints)
                joint_vel = robot.data.joint_vel  # shape: (num_envs, num_joints)
                joint_vel = robot.data.joint_vel  # shape: (num_envs, num_joints)
                
                # print for first environment only to avoid clutter

                # 获取当前的时间步
                if motion_command.time_steps[0] <= 3:  # print initial diagnostics
                    body_idx = robot.body_names.index("torso_link")
                    body_quat_env0 = robot.data.body_quat_w[0, body_idx].cpu().numpy()
                    print("="*60)
                    print("time_steps: ", motion_command.time_steps)
                    print("obs motion pos: ", obs[0, 0:20])
                    print("obs motion vel: ", obs[0, 20:40])
                    print("obs_quat: ", obs[0, 40:46])
                    print("obs_ang_vel: ", obs[0, 46:49])
                    # print("robot_ang_vel: ", robot.data.root_ang_vel_w)
                    print("obs_joint_pos: ", obs[0, 49:69])
                    print("obs_joint_vel: ", obs[0, 69:89])
                    print("last_action: ", obs[0, 89:109])
                    # print("robot body quat: ", body_quat_env0)
                    # print(f"[Step {timestep}] Robot joint_pos: {joint_pos[0].cpu().numpy()}")
                    # print(f"[Step {timestep}] Robot joint_vel: {joint_vel[0].cpu().numpy()}")
            except Exception:
                pass

            # agent stepping
            if args_cli.onnx_flag:
                time_steps = motion_command.time_steps.float().unsqueeze(1)
                obs_np = obs.detach().cpu().numpy()
                time_steps_np = time_steps.detach().cpu().numpy()

                outputs = session.run(
                    None,
                    {
                        "obs": obs_np,
                        "time_step": time_steps_np
                    }
                )
                actions = outputs[0]
                actions = torch.from_numpy(actions).float()
            else:
                actions = policy(obs)

            if motion_command.time_steps[0] <= 3:
                print("action: ", actions)
                print("="*60)           

            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
