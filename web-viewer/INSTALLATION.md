# Installation

From-scratch, repeatable setup for this repo. Primary path is fully dockerized: Isaac Sim
comes baked into Isaac Lab's own container image, and a separate lightweight container
serves a WebRTC viewer in the browser. A native (no-Docker) alternative is also documented.

## 1. System prerequisites

```bash
# Docker + NVIDIA Container Toolkit
sudo apt install docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

You also need an NVIDIA GPU + driver (see
[Isaac Sim technical requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)),
and [Tailscale](https://tailscale.com/) if you want to stream to a remote client over its
network (the examples below use `tailscale ip -4` to pick the advertised address — set
`ISAACSIM_HOST` yourself if you don't use Tailscale).

Sanity-check GPU access in Docker:

```bash
docker run --rm --runtime=nvidia --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 2. Clone

```bash
git clone --recurse-submodules <this-repo-url>
cd isaaclab_quadruped
# if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

## 3. Running (dockerized — primary path)

Everything Isaac Sim/Isaac Lab related runs inside Isaac Lab's own dev container. This repo
only adds a browser-based WebRTC viewer alongside it. This is the setup for a **headless
remote workstation you SSH into**.

### a) WebRTC viewer

```bash
ISAACSIM_HOST=$(tailscale ip -4) docker compose -f docker-compose.yml up --build -d
```

Independent of the Isaac Lab container — start/stop it any time, it just serves a page at
`http://<host-ip>:8210` that shows "WAITING FOR STREAM..." until something streams to it.
See [`DOCKER.md`](DOCKER.md) for ports, env vars, and troubleshooting.

### b) Isaac Lab dev container

Nothing auto-starts in this container — you get a bash shell and launch scripts yourself.
It bind-mounts `IsaacLab/source`, `scripts`, `docs`, `tools`, so code edits are
live with no rebuild. This is upstream Isaac Lab tooling (`container.py`), not anything
added by this repo:

```bash
cd IsaacLab/docker
./container.py start base    # builds once (pulls the Isaac Sim base image), idles at bash
./container.py enter base    # attach a shell
```

> **Headless / SSH, no local display:** the first `start` prompts for X11 (GUI) forwarding
> and saves your answer to `docker/.container.cfg` (gitignored, local-only). Answer no, or
> if it already errored with `KeyError: 'DISPLAY'`, edit that file and set
> `x11_forwarding_enabled = 0`. Not needed anyway since everything here runs `--headless`.

### c) Run something, streamed

From inside that shell, run whatever script you want, with `--livestream 2` to turn its
Isaac Sim process into a WebRTC source matching the viewer above:

```bash
# tmux/screen (or nohup ... &) so it survives you disconnecting your SSH session
tmux new -s train
python scripts/reinforcement_learning/rsl_rl/train.py --task <Task-Name> --headless --livestream 2
```

`isaac-lab-base` runs with `network_mode: host` (see
`IsaacLab/docker/docker-compose.yaml`), so it binds the same 49100/47998 ports
the viewer expects — no extra wiring needed.

Drop `--livestream 2` (keep `--headless`) to run off-screen with no viewer at all — e.g. for
unattended training runs where you don't need to watch.

**Teardown:**

```bash
./container.py stop base                        # from IsaacLab/docker
docker compose -f docker-compose.yml down        # from the repo root, stops the viewer
```

## 4. Running natively (no Docker, optional)

Lower latency on the machine that has the GPU, useful for quick local iteration without
container overhead. Not the primary path — most instructions above assume Docker.

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

# Isaac Sim, from pip
uv pip install "isaacsim[all,extscache]==6.0.0.0" --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow

# CUDA-enabled PyTorch matching that Isaac Sim build
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab itself, in editable mode, from the submodule
cd IsaacLab
./isaaclab.sh --install
cd ..
```

Verify:

```bash
./IsaacLab/isaaclab.sh -p -c "import isaaclab; print('ok')"
```

Run scripts the same way as in the container, just without `container.py`:

```bash
# without WebRTC — opens a normal on-screen window if a display is attached
./IsaacLab/isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# with WebRTC — pair with the viewer from step 3a
./IsaacLab/isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless --livestream 2
```
