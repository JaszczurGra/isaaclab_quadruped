import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Spawn the Spot robot from its local URDF into an empty scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils

# scripts/ and models/ are siblings at the repo root (mounted as quadruped_scripts/ and
# quadruped_scripts/../models in the Isaac Lab dev container).
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quadruped_models")
SPOT_URDF_PATH = os.path.join(MODELS_DIR, "spot.urdf")


def design_scene():
    """Designs the scene by spawning a ground plane, a light, and the Spot robot from its URDF."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn the Spot robot from its local URDF file (floating base, as on the real robot)
    cfg_spot = sim_utils.UrdfFileCfg(
        asset_path=SPOT_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        # resolves package://spot_description/meshes/... to models/meshes/...
        ros_package_paths=[{"name": "spot_description", "path": MODELS_DIR}],
    )
    cfg_spot.func("/World/Spot", cfg_spot, translation=(0.0, 0.0, 0.6))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 2.0, 1.2], [0.0, 0.0, 0.4])
    # Design scene
    design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
