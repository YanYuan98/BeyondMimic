"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

def mujoco_env():
    model_file = "/home/yyy/Documents/Work/Imitation/BeyondMimic/source/" \
    "whole_body_tracking/whole_body_tracking/assets/simple_model/xml/simple_model.xml"

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005

    dt = 0.005
    kp = 20.0
    kd = 1.0
    step_count = 0
    step_start = 0
    print("kp, kd: ", kp, kd)

    viewer = mujoco.viewer.launch_passive(model, data)    
    while viewer.is_running():
        time.sleep(0.01)

        target_pos = np.sin(2*np.pi*dt*step_count)

        current_pos = data.qpos.copy()
        current_vel = data.qvel.copy()

        print("time_num: ", step_count)
        print("dt: ", model.opt.timestep)
        print("time: ", data.time)
        print("robot target_pos: ", target_pos)
        print("robot qpos: ", data.qpos)
        print("robot qvel: ", data.qvel)

        torque = kp * (target_pos - current_pos) - kd * current_vel

        data.ctrl[:] = torque
        # data.qpos[0] = target_pos
        print("toruqe: ", data.ctrl)

        mujoco.mj_step(model, data)

        step_count += 1

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    pass

if __name__ == "__main__":

    ## run the main function
    mujoco_env()
