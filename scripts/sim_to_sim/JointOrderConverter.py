import numpy as np

class JointOrderConverter:
    """处理URDF和MuJoCo XML之间的joint顺序转换"""
    
    def __init__(self):
        # URDF中的joint顺序(网络输入/输出顺序)
        self.urdf_joint_order = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 
            'waist_pitch_joint',
            'left_hip_roll_joint', 'right_hip_roll_joint', 
            'waist_yaw_joint',
            'left_hip_yaw_joint', 'right_hip_yaw_joint', 
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 
            'left_knee_joint', 'right_knee_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
            'left_elbow_joint', 'right_elbow_joint',
            'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]
        
        # XML中的joint顺序(MuJoCo实际顺序)
        self.xml_joint_order = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_pitch_joint', 'waist_yaw_joint', 
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_joint'
        ]
        
        # 创建索引映射
        self._create_mapping()
    
    def _create_mapping(self):
        """创建URDF和XML之间的索引映射"""
        # 从XML顺序到URDF顺序的映射
        self.xml_in_urdf_indices = []
        for xml_joint in self.xml_joint_order:
            urdf_idx = self.urdf_joint_order.index(xml_joint)
            self.xml_in_urdf_indices.append(urdf_idx)
        # print("xml_in_urdf_indices: ", self.xml_in_urdf_indices)

        # 从URDF顺序到XML顺序的映射
        self.urdf_in_xml_indices = []
        for urdf_joint in self.urdf_joint_order:
            xml_idx = self.xml_joint_order.index(urdf_joint)
            self.urdf_in_xml_indices.append(xml_idx)
        # print("urdf_in_xml_indices: ", self.urdf_in_xml_indices)

        self.xml_in_urdf_indices = np.array(self.xml_in_urdf_indices)
        self.urdf_in_xml_indices = np.array(self.urdf_in_xml_indices)
    
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
            return urdf_ordered_data[self.xml_in_urdf_indices]
        else:
            # 批量数据
            return urdf_ordered_data[:, self.xml_in_urdf_indices]

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
            return xml_ordered_data[self.urdf_in_xml_indices]
        else:
            # 批量数据
            return xml_ordered_data[:, self.urdf_in_xml_indices]
