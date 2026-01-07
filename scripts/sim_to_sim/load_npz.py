import numpy as np

data = np.load("/home/yyy/Documents/Work/Imitation/BeyondMimic/artifacts/36_09_walk_forward_turn_back:v0/motion.npz")

print(data.files)   # 查看包含哪些数组

joint_pos = data["joint_pos"]
body_pos_w = data["body_pos_w"]
body_quat_w = data["body_quat_w"]
joint_vel = data["joint_vel"]
body_lin_vel_w = data["body_lin_vel_w"]
body_ang_vel_w = data["body_ang_vel_w"]

print("joint_pos", joint_pos[14])
print("joint_vel", joint_vel.shape)
print("body_quat_w", body_quat_w.shape)