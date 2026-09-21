# IsaacLab Quadruped

Quadruped robot RL work built on top of [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
(vendored as a submodule under `IsaacLab`), with a WebRTC viewer for watching
runs in a browser.

## Repo layout

| Path                  | What it is                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| `IsaacLab` | Isaac Lab framework, as a git submodule pinned to a specific upstream commit. Don't edit it in place — add your own tasks/extensions in this repo instead. Its own `docker/` folder (upstream, not ours) is how you actually run Isaac Sim/Isaac Lab, in a container. |
| `docker-compose.yml`, `web-viewer/` | A browser-based WebRTC viewer — pairs with the Isaac Lab container above, doesn't run Isaac Sim itself. |
| `.venv`                | Local Python env (managed with [uv](https://docs.astral.sh/uv/)) — only needed if you want to run Isaac Sim/Isaac Lab natively, without Docker. Optional; the primary workflow is fully dockerized. |

## Working in this repo

- **First time setup and how to run things** (dockerized — the primary path — and the native/no-Docker alternative): see [`INSTALLATION.md`](INSTALLATION.md).
- **WebRTC viewer reference** (ports, env vars, cloud deployment, troubleshooting): see [`DOCKER.md`](DOCKER.md).
- **Manual test procedures** for the viewer deployment: see [`TESTING.md`](TESTING.md).
- **Isaac Lab itself** (tasks, environments, training scripts, its own contribution guidelines, and its own Docker tooling): see `IsaacLab/README.md` and `IsaacLab/AGENTS.md`.

## Cloning

This repo uses a git submodule for Isaac Lab, so clone with:

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```









# starting

./container.py start base --files ../../docker-compose.scripts.yaml



## Entering the 

IsaacLab/docker/container.py enter base
tmux new -s train


python scripts/tutorials/00_sim/create_empty.py --headless


python scripts/demos/quadrupeds.py --livestream 2 --visualizer viser
 



# web-viewer is redunant



