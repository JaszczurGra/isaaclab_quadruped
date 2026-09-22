"""Minimal manager-based task: reward Spot for running forward as fast/far as possible.

Importing this module registers the gym id "Isaac-Spot-Forward-v0" as a side effect.

Unlike IsaacLab's own tuned ``Isaac-Velocity-Flat-Spot-v0`` task, this is deliberately
small: flat ground plane (no terrain generator/curriculum), no velocity-command tracking,
no per-foot gait shaping. It reuses Spot's own robot config (``SPOT_CFG``) and the
command-independent stability/energy penalty functions from IsaacLab's Spot mdp module so
the policy doesn't just learn to fling itself forward and faceplant.
"""

import gymnasium as gym
import torch

from isaaclab_physx.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_assets.robots.spot import SPOT_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

FORWARD_AXIS = 0  # body-frame index: 0 = x (forward), 1 = y (lateral), 2 = z (vertical)
MAX_REWARDED_SPEED = 4.0  # m/s; caps the per-step reward so a single physics spike can't be exploited

RUNS_ROOT = "runs"  # shared by train_spot_forward.py and play_spot_forward.py


def forward_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward forward (body-frame +x) linear velocity, clipped to a sane top speed."""
    asset = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b.torch[:, FORWARD_AXIS]
    return torch.clamp(forward_speed, min=-MAX_REWARDED_SPEED, max=MAX_REWARDED_SPEED)


@configclass
class SpotForwardSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene: terrain, the Spot robot, a contact sensor, and a light."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # covers every body so the "fell over" termination below can use it; air-time is not
    # tracked since this task doesn't use any per-foot gait-shaping rewards.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class SpotForwardActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class SpotForwardObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group. No velocity-command term: there is no command generator."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotForwardEventsCfg:
    """Reset/randomization events, borrowed from Spot's tuned locomotion task."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class SpotForwardRewardsCfg:
    """Reward terms for the MDP."""

    # -- task: the entire point of this environment
    forward_velocity = RewTerm(func=forward_velocity, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")})

    # -- stability/energy penalties, reused from Spot's tuned locomotion task so the policy
    # runs forward instead of flailing, flipping, or vibrating its joints to farm reward.
    base_motion = RewTerm(
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    joint_torques = RewTerm(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_vel = RewTerm(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )


@configclass
class SpotForwardTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )


@configclass
class SpotForwardEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the "run Spot forward" environment."""

    scene: SpotForwardSceneCfg = SpotForwardSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: SpotForwardObservationsCfg = SpotForwardObservationsCfg()
    actions: SpotForwardActionsCfg = SpotForwardActionsCfg()
    events: SpotForwardEventsCfg = SpotForwardEventsCfg()
    rewards: SpotForwardRewardsCfg = SpotForwardRewardsCfg()
    terminations: SpotForwardTerminationsCfg = SpotForwardTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10  # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.dt = 0.002  # 500 Hz physics, matching Spot's tuned task
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class SpotForwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO settings, copied from Spot's tuned runner cfg with wandb logging enabled."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "spot_forward"
    store_code_state = False
    logger = "wandb"
    wandb_project = "isaaclab-quadruped"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


gym.register(
    id="Isaac-Spot-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotForwardEnvCfg,
        "rsl_rl_cfg_entry_point": SpotForwardPPORunnerCfg,
    },
)
