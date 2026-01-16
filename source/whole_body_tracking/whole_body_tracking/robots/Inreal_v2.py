import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_L1 = 0.291
ARMATURE_L2 = 0.181
ARMATURE_L3 = 0.0509
ARMATURE_L4 = 0.001584
ARMATURE_A1 = ARMATURE_L2
ARMATURE_A2 = 0.036
ARMATURE_W_P = ARMATURE_L2
ARMATURE_W_Y = ARMATURE_A2


Inreal_V2_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/inreal_v2_description/urdf/inreal_v2.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.12),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.15,
            ".*_ankle_roll_joint": 0.0,
            ".*_elbow_joint": -0.5,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 586.85,
                ".*_hip_yaw_joint": 336.0,
                ".*_hip_roll_joint": 336.0,
                ".*_knee_joint": 270.0,
                ".*_ankle_pitch_joint": 270.0,
                ".*_ankle_roll_joint": 38.4,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 13.12,
                ".*_hip_yaw_joint": 12.9,
                ".*_hip_roll_joint": 12.9,
                ".*_knee_joint": 26.4,
                ".*_ankle_pitch_joint": 26.4,
                ".*_ankle_roll_joint": 40.75,
            },
            stiffness={
                ".*_hip_pitch_joint": 360,
                ".*_hip_roll_joint": 300,
                ".*_hip_yaw_joint": 300,
                ".*_knee_joint": 240,
                ".*_ankle_pitch_joint": 160,
                ".*_ankle_roll_joint": 40,
            },
            damping={
                ".*_hip_pitch_joint": 6.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*_ankle_pitch_joint": 4.0,
                ".*_ankle_roll_joint": 1.0,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_L1,
                ".*_hip_roll_joint": ARMATURE_L2,
                ".*_hip_yaw_joint": ARMATURE_L2,
                ".*_knee_joint": ARMATURE_L3,
                ".*_ankle_pitch_joint": ARMATURE_L3,
                ".*_ankle_roll_joint": ARMATURE_L4,
            },
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint", 
                "waist_pitch_joint"
                ],
            effort_limit_sim={
                "waist_yaw_joint": 140.25, 
                "waist_pitch_joint": 336,
                },
            velocity_limit_sim={
                "waist_yaw_joint": 26.82, 
                "waist_pitch_joint": 12.875,
                },
            stiffness={
                "waist_yaw_joint": 200, 
                "waist_pitch_joint": 300,
                },
            damping={
                "waist_yaw_joint": 4.0, 
                "waist_pitch_joint": 5.0,
                },
            armature={"waist_yaw_joint": ARMATURE_W_Y, 
                      "waist_pitch_joint": ARMATURE_W_P,
                },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 336.0,
                ".*_shoulder_roll_joint": 336.0,
                ".*_elbow_joint": 112.2,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.9,
                ".*_shoulder_roll_joint": 12.9,
                ".*_elbow_joint": 33.765,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 120,
                ".*_shoulder_roll_joint": 120,
                ".*_elbow_joint": 80,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_A1,
                ".*_shoulder_roll_joint": ARMATURE_A1,
                ".*_elbow_joint": ARMATURE_A2,
            },
        ),
    },
)

Inreal_V2_ACTION_SCALE = {}
for a in Inreal_V2_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            # Inreal_V2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
            Inreal_V2_ACTION_SCALE[n] = 0.25

# print("Inreal_V2_ACTION_SCALE:", Inreal_V2_ACTION_SCALE)
