"""Train Spot to run forward as far/fast as possible with RSL-RL.

    # local run, no wandb needed
    python quadruped_scripts/train_spot_forward.py --headless --logger tensorboard

    # a second run with different hyperparameters, kept in its own runs/ subfolder
    python quadruped_scripts/train_spot_forward.py --headless --logger tensorboard \\
        --run_name variant2 --learning_rate 5e-4 --entropy_coef 0.01

Results (checkpoints, tensorboard/wandb logs, dumped configs) are written under
RUNS_ROOT/<experiment_name>/<time-stamp>[_<run_name>]/. Defaults to wandb logging (see
spot_forward_env.SpotForwardPPORunnerCfg); pass --logger tensorboard to skip wandb login.
"""



# import os
# # Stop Omniverse from trying to launch the background hub
# os.environ["OMNI_CLIENT_ENABLE_HUB"] = "0"

# # (Optional) Force the carbonite logger to only show errors
# os.environ["CARB_LOG_LEVEL"] = "error"

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Spot to run forward with RSL-RL.")
parser.add_argument("--max_iterations", type=int, default=1000, help="Override the PPO runner's max_iterations.")
parser.add_argument(
    "--logger", type=str, default='wandb', choices=["tensorboard", "neptune", "wandb"], help="Override the RL logger."
)
parser.add_argument("--run_name", type=str, default="", help="Suffix appended to this run's log directory name.")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Override the PPO learning rate.")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Override the PPO entropy coefficient.")
parser.add_argument("--seed", type=int, default=42, help="Override the training seed.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=2048,
    help="Override the hardcoded NUM_ENVS. Escape hatch for running a second job on a GPU that's already busy.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import importlib.metadata as metadata
import os
from datetime import datetime

import gymnasium as gym
# from rsl_rl.runners import OnPolicyRunner
from train_spot_forward import OnPolicyRunner


from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from spot_forward_env import RUNS_ROOT, SpotForwardEnvCfg, SpotForwardPPORunnerCfg  # noqa: E402  (registers the task)

import wandb

INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

TASK_NAME = "Isaac-Spot-Forward-v0"


def main():
    """Main function."""

    num_envs = args_cli.num_envs 

    env_cfg = SpotForwardEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = SpotForwardPPORunnerCfg()
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed


    wandb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
    entity="isaaclab_doggo",
    # Set the wandb project where this run will be logged.
    project="test_spot_1",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "epochs": 1000,
    },
    )


    # migrate any deprecated rsl_rl cfg fields (e.g. legacy `stochastic`) to what the
    # installed rsl-rl-lib version actually expects
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)

    # specify directory for logging this run: RUNS_ROOT/<experiment_name>/<time-stamp>[_<run_name>]
    log_root_path = os.path.abspath(os.path.join(RUNS_ROOT, agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        log_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    print(f"[INFO] Training '{TASK_NAME}' with {num_envs} parallel environments, logger={agent_cfg.logger}.")
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(TASK_NAME, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,wandb_logger=wandb_logger)
    runner.add_git_repo_to_log(__file__)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"Gonna start the policy \n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    exit(0)
