# Installation for `ppo_mpc_training`

This project mixes **ROS 2** dependencies with a **Python RL stack**.
To keep installs reproducible, split them into two layers:

1. **ROS 2 dependencies** from `package.xml`
2. **Python dependencies** from `requirements.txt`

## 1. System assumptions

Recommended baseline:
- Ubuntu + ROS 2 environment already sourced
- Python **3.10+**
- Gazebo / TurtleBot3 simulation working

## 2. Install ROS 2 dependencies

From the repository root:

```bash
rosdep install --from-paths . --ignore-src -r -y
```

## 3. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install PyTorch first

Stable-Baselines3 depends on PyTorch, but the correct PyTorch wheel depends on
whether you use CPU or CUDA.

Use the official selector:

- https://pytorch.org/get-started/locally/

Examples:

### CPU-only
```bash
pip install torch torchvision torchaudio
```

### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.6
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 5. Install project Python dependencies

`requirements.txt` already keeps **NumPy on the 1.x line** for safer ROS 2 compatibility.

```bash
pip install -r requirements.txt
```

For development tools:

```bash
pip install -r requirements-dev.txt
```

## 6. Install the package itself in editable mode

```bash
pip install -e .
```

## 7. Validate the environment

```bash
python -m tb3_training.check_env --episodes 1 --max-steps 25
```

or, if your ROS 2 package entry points are available:

```bash
ros2 run tb3_training check_env --episodes 1 --max-steps 25
```

## 8. Start training the pure RL baseline

```bash
python -m tb3_training.tb3_train
```

or:

```bash
ros2 run tb3_training tb3_train
```

## Dependency policy

- Keep **ROS dependencies** in `package.xml`
- Keep **Python RL dependencies** in `requirements.in` / `requirements.txt`
- Keep **developer-only tools** in `requirements-dev.in` / `requirements-dev.txt`
- Do **not** pin PyTorch in the common requirements file; install it separately
  for the correct compute platform

## Recommended future improvement

For maximum reproducibility, add one of the following later:
- a Dockerfile
- a `Makefile` or `install.sh`
- separate lock files for CPU and CUDA environments
