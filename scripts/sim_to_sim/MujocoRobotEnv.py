import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from MotionObservation import MotionObservationTerm as ObsTerm
from JointOrderConverter import JointOrderConverter

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
        print("anchor index: ", self.robot_anchor_body_index[0], self.motion_anchor_body_index)

        self.converter = JointOrderConverter()
        # self.joint_index = self.get_joint_index()
        self.joint_names = self.get_joint_name()
        # print("关节名称: ", self.joint_names)
        self.action_scale = self.get_action_scale()
        self.action_scale_urdf = self.converter.xml_to_urdf(self.action_scale)
        print("关节scale xml: ", self.action_scale)
        print("关节scale urdf: ", self.action_scale_urdf)
        self.kp, self.kd = self.get_kp_kd()
        self.kp_urdf = self.converter.xml_to_urdf(self.kp)
        self.torque_clip = self.get_torque_clip()
        self.joint_pos_min, self.joint_pos_max = self.get_joint_pos_clip()
        print("关节kp xml: ", self.kp)
        print("关节kp urdf: ", self.kp_urdf)
        print("关节joint_pos_min: ", self.joint_pos_min)
        print("关节joint_pos_max: ", self.joint_pos_max)

        self.default_pos = self.get_default_pos()
        self.default_joint_pos = self.default_pos[7:]
        self.default_joint_pos_urdf = self.converter.xml_to_urdf(self.default_joint_pos)
        print("关节offset xml: ", self.default_joint_pos)
        print("关节offset urdf: ", self.default_joint_pos_urdf)

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

        # for i in range(self.model.nbody):
        #     print(self.model.body(i).name)

        return np.array(body_index)
    
    def get_joint_index(self):
        joint_index = []
        for joint_name, joint_cfg in self.cfg["joint"].items():
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            jnt_id = self.model.jnt_qposadr[index]
            joint_index.append(jnt_id)

        return joint_index
    
    def get_joint_name(self):
        joint_names = []
        print("njnt: ", self.model.njnt)
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name == None:
                print("none: ", jnt_type)
                pass
            else:
                joint_names.append(name)
                print(
                    f"joint_id = {i:02d}, "
                    f"name = {name:30s}"
                )

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
    
    def get_joint_pos_clip(self):
        joint_pos_min = []
        joint_pos_max = []
        for name in self.joint_names:
            if name not in self.cfg["Robot"]["joint"]:
                raise KeyError(f"Missing joint '{name}' in YAML config")
            joint_pos_min.append(self.cfg["Robot"]["joint"][name]["q_min"])
            joint_pos_max.append(self.cfg["Robot"]["joint"][name]["q_max"])

        return np.asarray(joint_pos_min), np.asarray(joint_pos_max)

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
        self.time_steps = 0
        mujoco.mj_resetData(self.model, self.data)

        obs_dummy = np.zeros((1, 109), dtype=np.float32)
        action, motion_state = self.get_session_output(obs_dummy)
        self.last_action = np.zeros(20, dtype=np.float32)

        # print("pos: ", self.data.qpos)
        # print("time: ", self.time_steps)

        root_pos = np.round(motion_state['body_pos_w'][0,0], 4)
        root_ori = np.round(motion_state['body_quat_w'][0,0], 4)
        joint_pos = np.round(motion_state['joint_pos'][0], 4)
        reset_joint_pos = self.converter.urdf_to_xml(joint_pos)

        root_lin_vel = np.round(motion_state['body_lin_vel_w'][0,0], 4)
        root_ang_vel = np.round(motion_state['body_ang_vel_w'][0,0], 4)
        root_ang_vel = np.array([0.0363, 0.0176, 0.1341])
        joint_vel = np.round(motion_state['joint_vel'][0], 4)
        reset_joint_vel = self.converter.urdf_to_xml(joint_vel)

        qpos = np.concatenate([
            root_pos,
            root_ori,
            reset_joint_pos
        ], axis=0)

        qvel = np.concatenate([
            root_lin_vel,
            root_ang_vel,
            reset_joint_vel
        ], axis=0)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
    
        mujoco.mj_forward(self.model, self.data)

        self.time_steps += 1

        print("init qpos", self.data.qpos)
        print("init qvel", self.data.qvel)
        
        return self.get_observation(motion_state)
    
    def get_session_output(self, obs):
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
        obs_dummy = np.zeros((1, 109), dtype=np.float32)
        _, motion_state = self.get_session_output(obs_dummy)
        print("time: ", self.time_steps)

        motion_anchor_quat_w = motion_state["body_quat_w"][0, self.motion_anchor_body_index]
        command_joint_pos, command_joint_vel = self.ObsTerm.get_command_obs(motion_state)
        # ensure command parts are numpy arrays with shape (1, N)
        command_joint_pos = np.asarray(command_joint_pos, dtype=np.float32)
        command_joint_vel = np.asarray(command_joint_vel, dtype=np.float32)
        command = np.concatenate([command_joint_pos, command_joint_vel], axis=1)

        motion_anchor_ori_b = self.ObsTerm.get_anchor_orientation_obs(motion_anchor_quat_w)
        base_ang_vel = np.asarray(self.ObsTerm.get_base_ang_vel_obs(), dtype=np.float32).reshape(-1)
        joint_pos_xml = np.asarray(self.ObsTerm.get_joint_pos_obs(), dtype=np.float32).reshape(-1)
        joint_vel_xml = np.asarray(self.ObsTerm.get_joint_vel_obs(), dtype=np.float32).reshape(-1)
        # convert MuJoCo (XML) joint ordering -> URDF (network) ordering for robot state terms
        joint_pos = self.converter.xml_to_urdf(joint_pos_xml)
        joint_vel = self.converter.xml_to_urdf(joint_vel_xml)

        last_action = np.asarray(self.ObsTerm.get_last_action_obs(self.last_action), dtype=np.float32).reshape(-1)

        # print("time: ", self.time_steps)
        # print("motion: ", command)
        # print("motion_anchor_ori_b: ", motion_anchor_ori_b)
        # print("base_ang_vel: ", base_ang_vel)
        # print("joint_pos: ", joint_pos)
        # print("joint_vel: ", joint_vel)
        # print("joint_pos_xml: ", joint_pos_xml)
        # print("joint_vel_xml: ", joint_vel_xml)
        # print("last_action: ", last_action)

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
        target_pos = np.clip(
            target_pos,
            self.joint_pos_min,
            self.joint_pos_max
        )
        target_pos_urdf = self.converter.xml_to_urdf(target_pos)

        print("target_pos: ", target_pos_urdf)
        
        # PD控制
        for _ in range(self.decimation):
            current_pos = self.data.qpos[7:]
            current_vel = self.data.qvel[6:]
            
            print("*"*60)
            # print("dt: ", self.model.opt.timestep)
            # print("sim time: ", self.data.time)
            # print("current_pos: ", current_pos)
            print("current_pos urdf: ", self.converter.xml_to_urdf(current_pos))
            print("current_vel urdf: ", self.converter.xml_to_urdf(current_vel))
            torque = self.kp * (target_pos - current_pos) - self.kd * current_vel
            torque = np.clip(torque, -self.torque_clip, self.torque_clip)
            torque_urdf = self.converter.xml_to_urdf(torque)
            print("torque: ", torque)
            print("torque urdf: ", torque_urdf)

            # 应用力矩
            self.data.ctrl[:] = torque
            print("torque: ", self.data.ctrl[:])
        
            # 前进仿真
            mujoco.mj_step(self.model, self.data)
            # mujoco.mj_rnePostConstraint(self.model, self.data)

        
        # 获取新观测
        self.time_steps += 1
        obs_dummy = np.zeros((1, 109), dtype=np.float32)
        _, motion_state = self.get_session_output(obs_dummy)
        obs = self.get_observation(motion_state)
        
        # 简单的终止条件：机器人翻倒
        base_height = self.data.qpos[2]
        done = base_height < 1.0  # 根据实际机器人调整
        
        reward = 0.0  # MuJoCo测试不需要奖励
        info = {}
        
        return obs, reward, done, info
