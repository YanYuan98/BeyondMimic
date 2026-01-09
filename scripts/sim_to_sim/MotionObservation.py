import torch
import numpy as np
from math_utlis import matrix_from_quat, subtract_frame_transforms, quat_inv, quat_mul


class MotionObservationTerm:
    def __init__(self, robot_anchor_body_index, default_joint_pos, mj_data, mj_model):
        self.default_joint_pos = default_joint_pos
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.robot_anchor_body_index = robot_anchor_body_index
        pass

    def get_command_obs(self, motion_state):
        joint_pos = motion_state["joint_pos"]
        joint_vel = motion_state["joint_vel"]
        return joint_pos, joint_vel

    def get_anchor_orientation_obs(self, motion_anchor_quat_w):
        # ensure inputs are torch tensors on CPU
        robot_q = torch.from_numpy(np.array(self.mj_data.xquat[self.robot_anchor_body_index])).float()

        if motion_anchor_quat_w is None:
            # use inverse of robot anchor as fallback
            ori = quat_inv(robot_q)
        else:
            # motion_anchor_quat_w may be numpy array or torch tensor
            if isinstance(motion_anchor_quat_w, np.ndarray):
                motion_q = torch.from_numpy(motion_anchor_quat_w).float()
            elif isinstance(motion_anchor_quat_w, torch.Tensor):
                motion_q = motion_anchor_quat_w.float()
            else:
                motion_q = torch.tensor(motion_anchor_quat_w, dtype=torch.float32)

            ori = quat_mul(quat_inv(robot_q), motion_q)
            # print("ori:", ori)
            # print("robot_q: ", robot_q)
            # print("quat_inv(robot_q):", quat_inv(robot_q))
            # print("motion_q:", motion_q)
            # print("ori:", ori)

        mat = matrix_from_quat(ori)

        # print("mat:", mat)


        # matrix_from_quat may return (N,3,3) for batched input or (3,3) for single
        if mat.ndim == 3:
            res = mat[..., :2].reshape(mat.shape[0], -1).detach().cpu().numpy()
        else:
            res = mat[..., :2].reshape(-1).detach().cpu().numpy()

        return res

    def get_base_ang_vel_obs(self):
        return np.array(self.mj_data.qvel[3:6], dtype=np.float32)

    def get_joint_pos_obs(self):
        qpos = np.array(self.mj_data.qpos[7:], dtype=np.float32)
        default = np.array(self.default_joint_pos, dtype=np.float32)
        return qpos - default

    def get_joint_vel_obs(self):
        return np.array(self.mj_data.qvel[6:], dtype=np.float32)

    def get_last_action_obs(self, last_action):
        return last_action
    