import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_L1 = 0.217
ARMATURE_L2 = 0.181
ARMATURE_L3 = 0.0512
ARMATURE_L4 = 0.00472
ARMATURE_A1 = ARMATURE_L3
ARMATURE_A2 = 0.036
ARMATURE_W = ARMATURE_L3

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_L1 = ARMATURE_L1 * NATURAL_FREQ**2
STIFFNESS_L2 = ARMATURE_L2 * NATURAL_FREQ**2
STIFFNESS_L3 = ARMATURE_L3 * NATURAL_FREQ**2
STIFFNESS_L4 = ARMATURE_L4 * NATURAL_FREQ**2
STIFFNESS_A1 = ARMATURE_A1 * NATURAL_FREQ**2
STIFFNESS_A2 = ARMATURE_A2 * NATURAL_FREQ**2
STIFFNESS_W = ARMATURE_W * NATURAL_FREQ**2

DAMPING_L1 = 2.0 * DAMPING_RATIO * ARMATURE_L1 * NATURAL_FREQ
DAMPING_L2 = 2.0 * DAMPING_RATIO * ARMATURE_L2 * NATURAL_FREQ
DAMPING_L3 = 2.0 * DAMPING_RATIO * ARMATURE_L3 * NATURAL_FREQ
DAMPING_L4 = 2.0 * DAMPING_RATIO * ARMATURE_L4 * NATURAL_FREQ
DAMPING_A1 = 2.0 * DAMPING_RATIO * ARMATURE_A1 * NATURAL_FREQ
DAMPING_A2 = 2.0 * DAMPING_RATIO * ARMATURE_A2 * NATURAL_FREQ
DAMPING_W = 2.0 * DAMPING_RATIO * ARMATURE_W * NATURAL_FREQ

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
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 495.0,
                ".*_hip_yaw_joint": 336.0,
                ".*_hip_roll_joint": 336.0,
                ".*_knee_joint": 342.0,
                ".*_ankle_pitch_joint": 342.0,
                ".*_ankle_roll_joint": 33.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 15.6,
                ".*_hip_yaw_joint": 12.9,
                ".*_hip_roll_joint": 12.9,
                ".*_knee_joint": 24.4,
                ".*_ankle_pitch_joint": 24.4,
                ".*_ankle_roll_joint": 34.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_L1,
                ".*_hip_roll_joint": STIFFNESS_L2,
                ".*_hip_yaw_joint": STIFFNESS_L2,
                ".*_knee_joint": STIFFNESS_L3,
                ".*_ankle_pitch_joint": STIFFNESS_L3,
                ".*_ankle_roll_joint": STIFFNESS_L4,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_L1,
                ".*_hip_roll_joint": DAMPING_L2,
                ".*_hip_yaw_joint": DAMPING_L2,
                ".*_knee_joint": DAMPING_L3,
                ".*_ankle_pitch_joint": DAMPING_L3,
                ".*_ankle_roll_joint": DAMPING_L4,
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
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=336,
            velocity_limit_sim=12.9,
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_W,
            damping=2.0 * DAMPING_W,
            armature=2.0 * ARMATURE_W,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 336.0,
                ".*_shoulder_roll_joint": 336.0,
                ".*_elbow_joint": 136.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 12.9,
                ".*_shoulder_roll_joint": 12.9,
                ".*_elbow_joint": 27.5,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_A1,
                ".*_shoulder_roll_joint": STIFFNESS_A1,
                ".*_elbow_joint": STIFFNESS_A2,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_A1,
                ".*_shoulder_roll_joint": DAMPING_A1,
                ".*_elbow_joint": DAMPING_A2,
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
            Inreal_V2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
