"""Script to test Isaac Lab trained policy in MuJoCo simulator (sim-to-sim transfer)."""

import argparse
import os
import yaml
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
import onnxruntime as ort
from MotionObservation import MotionObservationTerm as ObsTerm

class JointOrderConverter:
    """处理URDF和MuJoCo XML之间的joint顺序转换"""
    
    def __init__(self):
        # URDF中的joint顺序(网络输入/输出顺序)
        self.urdf_joint_order = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_pitch_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_yaw_joint',
            'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint', 'left_knee_joint', 'right_knee_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_pitch_joint',
            'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]
        
        # XML中的joint顺序(MuJoCo实际顺序)
        self.xml_joint_order = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_pitch_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint', 'right_elbow_joint'
        ]
        
        # 创建索引映射
        self._create_mapping()
    
    def _create_mapping(self):
        """创建URDF和XML之间的索引映射"""
        # 从XML顺序到URDF顺序的映射
        self.xml_to_urdf_indices = []
        for xml_joint in self.xml_joint_order:
            urdf_idx = self.urdf_joint_order.index(xml_joint)
            self.xml_to_urdf_indices.append(urdf_idx)
        
        # 从URDF顺序到XML顺序的映射
        self.urdf_to_xml_indices = []
        for urdf_joint in self.urdf_joint_order:
            xml_idx = self.xml_joint_order.index(urdf_joint)
            self.urdf_to_xml_indices.append(xml_idx)
        
        self.xml_to_urdf_indices = np.array(self.xml_to_urdf_indices)
        self.urdf_to_xml_indices = np.array(self.urdf_to_xml_indices)
    
    def xml_to_urdf(self, xml_ordered_data):
        """
        将XML顺序(MuJoCo获取的数据)转换为URDF顺序(网络需要的输入)
        
        参数:
            xml_ordered_data: 按XML顺序排列的joint数据(position/velocity等)
                            shape: (20,) 或 (batch_size, 20)
        
        返回:
            urdf_ordered_data: 按URDF顺序排列的数据
        """
        xml_ordered_data = np.array(xml_ordered_data)
        
        if xml_ordered_data.ndim == 1:
            # 单个样本
            return xml_ordered_data[self.xml_to_urdf_indices]
        else:
            # 批量数据
            return xml_ordered_data[:, self.xml_to_urdf_indices]
    
    def urdf_to_xml(self, urdf_ordered_data):
        """
        将URDF顺序(网络输出的action)转换为XML顺序(MuJoCo需要的控制输入)
        
        参数:
            urdf_ordered_data: 按URDF顺序排列的数据(网络输出的action)
                             shape: (20,) 或 (batch_size, 20)
        
        返回:
            xml_ordered_data: 按XML顺序排列的数据
        """
        urdf_ordered_data = np.array(urdf_ordered_data)
        
        if urdf_ordered_data.ndim == 1:
            # 单个样本
            return urdf_ordered_data[self.urdf_to_xml_indices]
        else:
            # 批量数据
            return urdf_ordered_data[:, self.urdf_to_xml_indices]
    
    def verify_mapping(self):
        """验证映射是否正确"""
        print("验证映射关系:")
        print("-" * 60)
        for i, urdf_joint in enumerate(self.urdf_joint_order):
            xml_idx = self.urdf_to_xml_indices[i]
            xml_joint = self.xml_joint_order[xml_idx]
            print(f"URDF[{i:2d}] {urdf_joint:30s} -> XML[{xml_idx:2d}] {xml_joint}")
        
        # 验证双向映射一致性
        test_data = np.arange(20)
        converted = self.urdf_to_xml(test_data)
        recovered = self.xml_to_urdf(converted)
        assert np.allclose(test_data, recovered), "映射验证失败!"
        print("-" * 60)
        print("✓ 映射验证通过!")

class MuJoCoRobotEnv:
    """MuJoCo机器人仿真环境包装器"""
    
    def __init__(
        self,
        model_path: str,
        policy_path: str,
        cfg: dict,
        body_names: list,
        anchor_body_name: str,
        render: bool = True,
        dt: float = 0.005,  # 50Hz control frequency
    ):
        """
        Args:
            model_path: MuJoCo XML模型文件路径
            policy_path: ONNX策略文件路径
            metadata_path: 元数据JSON文件路径（包含观测归一化参数等）
            render: 是否渲染可视化
            dt: 控制时间步长
        """
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.decimation = 4
        self.cfg = cfg
        self.time_steps = 0

        self.body_names = body_names
        self.anchor_body_name = anchor_body_name
        self.body_indexes = self.get_body_index(self.body_names)
        self.robot_anchor_body_index = self.get_body_index([self.anchor_body_name])
        self.motion_anchor_body_index = self.body_names.index(self.anchor_body_name)
        print("anchor index: ", self.robot_anchor_body_index, self.motion_anchor_body_index)

        self.converter = JointOrderConverter()
        # self.joint_index = self.get_joint_index()
        self.joint_names = self.get_joint_name()
        print("关节名称: ", self.joint_names)
        self.action_scale = self.get_action_scale()
        self.kp, self.kd = self.get_kp_kd()
        self.torque_clip = self.get_torque_clip()

        self.default_pos = self.get_default_pos()
        self.default_joint_pos = self.default_pos[7:]

        self.ObsTerm = ObsTerm(self.robot_anchor_body_index[0], self.default_joint_pos, self.data, self.model)
        
        # 加载策略
        self.session = ort.InferenceSession(policy_path)
        
        self.render_mode = render
        self.viewer = None
        
        # 机器人状态
        self.num_dof = self.model.nv  # 自由度数量
        self.last_action = np.zeros(20, dtype=np.float32)
        
        print(f"[INFO] MuJoCo环境初始化完成")
        print("[INFO] 自由度数量: ", self.num_dof)
    
    def get_body_index(self, body_names):
        body_index = []
        for name in body_names:
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            body_index.append(index)

        return torch.tensor(body_index, dtype=torch.long)
    
    def get_joint_index(self):
        joint_index = []
        for joint_name, joint_cfg in self.cfg["joint"].items():
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            jnt_id = self.model.jnt_qposadr[index]
            joint_index.append(jnt_id)

        return joint_index
    
    def get_joint_name(self):
        joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name == None or name == "floating_base":
                pass
            else:
                joint_names.append(name)

        return np.asarray(joint_names)
    
    def get_default_pos(self):
        default_pos = []
        default_pos += self.cfg["Robot"]["default_pos"]["base_pos"]
        default_pos += self.cfg["Robot"]["default_pos"]["base_ori"]
        for name in self.joint_names:
            if name not in self.cfg["Robot"]["joint"]:
                raise KeyError(f"Missing joint '{name}' in YAML config")
            default_pos.append(self.cfg["Robot"]["joint"][name]["default_joint_pos"])

        return np.asarray(default_pos)
    
    def get_action_scale(self):
        action_scale = []
        for name in self.joint_names:
            if name not in self.cfg["Robot"]["joint"]:
                raise KeyError(f"Missing joint '{name}' in YAML config")
            action_scale.append(self.cfg["Robot"]["joint"][name]["action_scale"])

        return np.asarray(action_scale)
    
    def get_torque_clip(self):
        torque_clip = []
        for name in self.joint_names:
            if name not in self.cfg["Robot"]["joint"]:
                raise KeyError(f"Missing joint '{name}' in YAML config")
            torque_clip.append(self.cfg["Robot"]["joint"][name]["clip"])

        return np.asarray(torque_clip)
    
    def get_kp_kd(self):
        kp = []
        kd = []
        for name in self.joint_names:
            if name not in self.cfg["Robot"]["joint"]:
                raise KeyError(f"Missing joint '{name}' in YAML config")
            kp.append(self.cfg["Robot"]["joint"][name]["stiffness"])
            kd.append(self.cfg["Robot"]["joint"][name]["damping"])

        return np.asarray(kp), np.asarray(kd)

    def reset(self) -> np.ndarray:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 添加小的随机扰动
        # self.data.qpos[:] = self.default_pos + np.random.randn(len(self.default_pos)) * 0.1
        self.data.qpos[:] = self.default_pos
        self.data.qvel[:] = np.random.randn(self.model.nv) * 0.1
        
        mujoco.mj_forward(self.model, self.data)

        obs_dummy = np.zeros((1, 109), dtype=np.float32)
        action, motion_state = self.get_session_output(obs_dummy)

        print("pos: ", self.data.qpos)
        print("time: ", self.time_steps)
        
        return self.get_observation(motion_state)
    
    def get_session_output(self, obs):
        print("input obs: ", obs.shape)
        print("input time: ", self.time_steps)
        outputs = self.session.run(
            None,
            {
                "obs": obs,
                "time_step": np.array([[self.time_steps]], dtype=np.float32)
            }
        )
        motion_state = {
            "joint_pos": outputs[1],
            "joint_vel": outputs[2],
            "body_pos_w": outputs[3],
            "body_quat_w": outputs[4],
            "body_lin_vel_w": outputs[5],
            "body_ang_vel_w": outputs[6]
        }
        action = outputs[0]

        return action, motion_state
        

    def get_observation(self, motion_state) -> np.ndarray:
        """构建观测向量
        
        典型的观测包括：
        - 基座线速度 (3)
        - 基座角速度 (3)
        - 投影重力向量 (3)
        - 关节位置 (num_dof)
        - 关节速度 (num_dof)
        - 上一步动作 (num_dof)
        """
        
        # command
        # motion_state outputs are already in the network (URDF) order
        motion_anchor_quat_w = motion_state["body_quat_w"][0, self.motion_anchor_body_index]
        command_joint_pos, command_joint_vel = self.ObsTerm.get_command_obs(motion_state)
        # ensure command parts are numpy arrays with shape (1, N)
        command_joint_pos = np.asarray(command_joint_pos, dtype=np.float32)
        command_joint_vel = np.asarray(command_joint_vel, dtype=np.float32)
        command = np.concatenate([command_joint_pos, command_joint_vel], axis=1)
        # command_joint_pos_urdf = self.converter.xml_to_urdf(command_joint_pos)
        # command_joint_vel_urdf = self.converter.xml_to_urdf(command_joint_vel)
        # command = np.concatenate([command_joint_pos_urdf, command_joint_vel_urdf], axis=1)

        motion_anchor_ori_b = self.ObsTerm.get_anchor_orientation_obs(motion_anchor_quat_w)
        base_ang_vel = np.asarray(self.ObsTerm.get_base_ang_vel_obs(), dtype=np.float32).reshape(-1)
        joint_pos = np.asarray(self.ObsTerm.get_joint_pos_obs(), dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(self.ObsTerm.get_joint_vel_obs(), dtype=np.float32).reshape(-1)
        # convert MuJoCo (XML) joint ordering -> URDF (network) ordering for robot state terms
        joint_pos = self.converter.xml_to_urdf(joint_pos)
        joint_vel = self.converter.xml_to_urdf(joint_vel)

        last_action = np.asarray(self.ObsTerm.get_last_action_obs(self.last_action), dtype=np.float32).reshape(-1)
        # print("command_bef: ", command_befx)
        print("command: ", command)
        print("motion_anchor_ori_b: ", motion_anchor_ori_b)
        print("base_ang_vel: ", base_ang_vel)
        print("joint_pos: ", joint_pos)
        print("joint_vel: ", joint_vel)
        print("last_action: ", last_action)

        obs_terms = [
            np.asarray(command, dtype=np.float32).reshape(-1),
            np.asarray(motion_anchor_ori_b, dtype=np.float32).reshape(-1),
            np.asarray(base_ang_vel, dtype=np.float32).reshape(-1),
            np.asarray(joint_pos, dtype=np.float32).reshape(-1),
            np.asarray(joint_vel, dtype=np.float32).reshape(-1),
            np.asarray(last_action, dtype=np.float32).reshape(-1),
        ]

        obs_terms = [
            np.asarray(term, dtype=np.float32).reshape(-1)
            for term in obs_terms
        ]

        obs = np.concatenate(obs_terms, axis=0)
        self.time_steps += 1
        
        return obs
    
    def step(self, action: np.ndarray) -> tuple:
        """执行一步仿真
        
        Args:
            action: 策略输出的动作（通常是目标关节位置的残差）
        
        Returns:
            observation, reward, done, info
        """
        # 保存动作用于下一次观测
        self.last_action = action.copy()

        action_xml = self.converter.urdf_to_xml(action)
        
        
        # 计算目标关节位置
        target_pos = self.default_joint_pos + action_xml * self.action_scale
        
        # PD控制
        for _ in range(self.decimation):
            current_pos = self.data.qpos[7:7+self.num_dof]
            current_vel = self.data.qvel[6:6+self.num_dof]
            
            torque = self.kp * (target_pos - current_pos) - self.kd * current_vel
            
            # 应用力矩
            self.data.ctrl[:] = np.clip(torque, -self.torque_clip, 
                                        self.torque_clip)
        
            # 前进仿真
            mujoco.mj_step(self.model, self.data)
        
        # 获取新观测
        obs_dummy = np.zeros((1, 109), dtype=np.float32)
        _, motion_state = self.get_session_output(obs_dummy)
        obs = self.get_observation(motion_state)
        
        # 简单的终止条件：机器人翻倒
        base_height = self.data.qpos[2]
        done = base_height < 0.8  # 根据实际机器人调整
        
        reward = 0.0  # MuJoCo测试不需要奖励
        info = {}
        
        return obs, reward, done, info
    
    def render(self):
        """渲染可视化"""
        if self.render_mode and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer is not None:
            self.viewer.sync()


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
        # model_file = "/home/yyy/Documents/Work/Imitation/BeyondMimic/source/" \
        # "whole_body_tracking/whole_body_tracking/assets/inreal_v2_description/urdf/inreal_v2.urdf"
        cfg_file = "/home/yyy/Documents/Work/Imitation/BeyondMimic/scripts/sim_to_sim/default_cfg.yaml"
    
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    print("="*60)
    print("MuJoCo Sim-to-Sim 测试")
    print("="*60)
    print(f"模型文件: {model_file}")
    print(f"策略文件: {args.policy}")
    print(f"测试回合: {args.episodes}")
    print("="*60)

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
    )
    
    # 运行测试
    render = True
    episode_lengths = []
    
    for episode in range(args.episodes):
        print(f"\n[Episode {episode+1}/{args.episodes}]")
        
        obs = env.reset()
        done = False
        step_count = 0

        # break
        
        while not done and step_count < args.max_steps:
            time.sleep(0.5)  # 控制频率
            # 推理动作
            obs_tensor = obs.reshape(1, -1).astype(np.float32)
            action, _ = env.get_session_output(obs_tensor)
            action = action.flatten()
            
            # 执行动作
            obs, _, done, _ = env.step(action)
            
            # 渲染
            if render:
                env.render()
            
            step_count += 1
        
        episode_lengths.append(step_count)
        print(f"Episode length: {step_count} steps ({step_count * 0.005:.2f}s)")
        
        if done:
            print("Episode terminated early (robot fell)")
    
    # 统计信息
    print("\n" + "="*60)
    print("测试统计")
    print("="*60)
    print(f"平均episode长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"成功率: {sum(l >= args.max_steps for l in episode_lengths) / len(episode_lengths) * 100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()