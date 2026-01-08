import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.actuators import DelayedDCMotorCfg

BLACKPANTHER_CFG = ArticulationCfg( # TODO: Not Tested Yet
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/mirrorme/bp2/urdf/bp21spring.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            'FL_hip_joint': 0.1,  # [rad]
            'FL_thigh_joint': -0.52,  # [rad]
            'FL_calf_joint': 1.04,  # [rad]

            'FR_hip_joint': -0.1,  # [rad]
            'FR_thigh_joint': -0.52,  # [rad]
            'FR_calf_joint': 1.04,  # [rad]

            'RL_hip_joint': 0.1,  # [rad]
            'RL_thigh_joint': 0.52,  # [rad]
            'RL_calf_joint': -1.04,  # [rad]~

            'RR_hip_joint': -0.1,  # [rad]
            'RR_thigh_joint': 0.52,  # [rad]
            'RR_calf_joint': -1.04,  # [rad]

            'toe_rl_j': 0.0,   # [rad]
            'toe_rr_j': 0.0,   # [rad]
            'toe_fl_j': 0.0,   # [rad]
            'toe_fr_j': 0.0,   # [rad]
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        # Hip and Thigh (abduction/adduction + rotation)
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
            effort_limit=102.0,
            velocity_limit=41.0,
            stiffness=25.0,
            damping=1.5,
            friction=0.0,
            min_delay=0,
            max_delay=4,  # up to 25ms @ 200Hz
        ),

        # Knee joints
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_to_knee_.*_j"],
            effort_limit=80.0,
            velocity_limit=60.0,
            stiffness=30.0,
            damping=1.5,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        # Toe prismatic joints (spring-loaded toe actuators)
        "ToeSpring": DelayedPDActuatorCfg(
            joint_names_expr=[".*toe_.*_j"],
            effort_limit=100.0,
            velocity_limit=5.0,
            stiffness=6000.0,  # matches urdf toe spring
            damping=30.0,
            friction=0.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)

INREAL2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/mirrorme/Inrealv2/blackman_pelvis.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),  # 站立高度
        joint_pos= { # = target angles [rad] when action = 0.0
            "hip_pitch_L" : -0.1,
            "hip_roll_L" : 0. ,
            "hip_yaw_L" : 0. ,
            "knee_L" : 0.3,
            "ankle_pitch_L" : -0.2,
            "ankle_roll_L" : 0,
            "hip_pitch_R" : -0.1,
            "hip_roll_R" : 0. ,
            "hip_yaw_R" : 0. ,
            "knee_R" : 0.3,
            "ankle_pitch_R" : -0.2,
            "ankle_roll_R" : 0,
            "waist_pitch" : 0,
            "waist_yaw" : 0,
            "shoulder_pitch_L" : 0,
            "shoulder_roll_L" : 0.25,
            "elbow_L" : -0.4,
            "shoulder_pitch_R" : 0,
            "shoulder_roll_R" : -0.25,
            "elbow_R" : -0.4,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        "HipPitch": DelayedDCMotorCfg(
            joint_names_expr=["hip_pitch_.*"],
            effort_limit=495.0,
            velocity_limit=15.6,
            critical_speed=6.8,
            armature=0.217,
            stiffness=360.0,
            damping=6.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),


        "HipRoll": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["hip_roll_.*"],
            effort_limit=336.0,
            velocity_limit=12.9,
            critical_speed=6.8,
            armature=0.181,
            stiffness=300.0,
            damping=5.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "HipYaw": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["hip_yaw_.*"],
            effort_limit=336.0,
            velocity_limit=12.9,
            critical_speed=6.8,
            armature=0.181,
            stiffness=300.0,
            damping=5.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "Knee": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["knee_.*"],
            effort_limit=270.0,
            velocity_limit=26.4,
            critical_speed=18.0,
            armature=0.0512,
            stiffness=240.0,
            damping=5.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "AnklePitch": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["ankle_pitch_.*"],
            effort_limit=270.0,
            velocity_limit=26.4,
            critical_speed=18.0,
            armature=0.0512,
            stiffness=160.0,
            damping=4.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "AnkleRoll": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["ankle_roll_.*"],
            effort_limit=33.0,
            velocity_limit=34.0,
            critical_speed=16.9,
            armature=0.00472,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "Waist": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["waist_.*"],
            effort_limit=336.0,
            velocity_limit=12.9,
            critical_speed=6.8,
            armature=0.181,
            stiffness=300.0,
            damping=5.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "Shoulder": DelayedDCMotorCfg(  # 改为DelayedDCMotorCfg
            joint_names_expr=["shoulder_.*"],
            effort_limit=336.0,
            velocity_limit=12.9,
            critical_speed=6.8,
            armature=0.181,
            stiffness=120.0,
            damping=3.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),

        "Elbow": DelayedDCMotorCfg(
            joint_names_expr=["elbow_.*"],
            effort_limit=136.0,
            velocity_limit=27.5,
            critical_speed=18.4,
            armature=0.036,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
    },
)