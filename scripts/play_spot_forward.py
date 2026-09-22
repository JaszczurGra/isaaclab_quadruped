"""Run a trained Spot-forward checkpoint in the simulator.

    python quadruped_scripts/play_spot_forward.py
    $ python quadruped_scripts/play_spot_forward.py --checkpoint runs/spot_forward/2026-09-22_00-45-47/model_950.pt 
    python quadruped_scripts/play_spot_forward.py --num_envs 16 --headless --livestream 2

With no --checkpoint, resolves the latest checkpoint from the most recent run under
RUNS_ROOT (matching train_spot_forward.py's output directory).
"""





import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Spot-forward RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to a specific model_*.pt checkpoint. Defaults to the latest."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""





import importlib.metadata as metadata
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

from spot_forward_env import RUNS_ROOT, SpotForwardEnvCfg, SpotForwardPPORunnerCfg  # noqa: E402  (registers the task)

TASK_NAME = "Isaac-Spot-Forward-v0"
INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_visualizers.rerun import RerunVisualizerCfg


visualizer_cfg = RerunVisualizerCfg(
    app_id="isaaclab-simulation",             
    grpc_port=9875,              # Safe data stream port (no 9090 conflict!)
    web_port=9091,               # Browser user interface server port
    bind_address="0.0.0.0",      # Broad binding to allow Tailscale incoming traffic
    open_browser=False,          # Stop headless server from failing to open a browser window
    keep_historical_data=False,               
    keep_scalar_history=False,                
    record_to_rrd=None          
)

# Attach it to the main Physics/Simulation configuration
sim_cfg = SimulationCfg(
    dt=1 / 120, 
    # render_fps=30,
    visualizer_cfgs=[visualizer_cfg] # Pass your custom Rerun config array here
)





def main():



    agent_cfg = SpotForwardPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join(RUNS_ROOT, agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env_cfg = SpotForwardEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.visualizer_cfgs = [visualizer_cfg]
    # a fallen robot shouldn't end the episode while we're just watching it run
    env_cfg.terminations.body_contact = None

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
