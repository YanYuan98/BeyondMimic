"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--robot_name", type=str, default="simple", choices=["simple"])

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.simple_model import Simple_CYLINDER_CFG


# -------------------------- enviroment setting --------------------------
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    if args_cli.robot_name=="simple":
        print("[Robot] simple robot!")
        robot: ArticulationCfg = Simple_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # Extract scene entities
    robot = scene["robot"]

    print("kp, kd: ", robot.data.joint_stiffness, robot.data.joint_damping)

    # Simulation loop
    num = 0
    dt = sim.get_physics_dt()
    while simulation_app.is_running():
        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[0] = np.sin(2*np.pi*dt*num)
        joint_vel[0] = np.cos(2*np.pi*dt*num)

        print("time_num: ", num)
        print("dt: ", sim.get_physics_dt())
        print("time: ", dt*num)
        print("target qpos: ", joint_pos)
        print("robot qpos: ", robot.data.joint_pos[0])
        print("robot qvel: ", robot.data.joint_vel[0])

        robot.set_joint_position_target(joint_pos, joint_ids=[0])
        scene.write_data_to_sim()
        # sim.render()  # We don't want physic (sim.step())
        sim.step(render=True)
        scene.update(sim.get_physics_dt())

        num += 1
        pos_lookat = np.array([0.0, 0.0, 0.0])
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def isaaclab_env():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.005
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
    )

if __name__ == "__main__":

    ## run the main function
    isaaclab_env()

    # close sim app
    simulation_app.close()
