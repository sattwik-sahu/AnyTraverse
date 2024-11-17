# CLIPSEG Offroad Navigation

For Vision-Language Based Offroad Navigation.

## Installation

Follow the steps below to set up the project in a Python virtual environment, install Poetry, and configure PyTorch based on your system requirements.

### 1. Set Up a Python Virtual Environment

First, create and activate a Python virtual environment to isolate the project dependencies:

**On Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Poetry

If Poetry is not installed on your system, you can install it by following the official instructions:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, ensure Poetry is in your system's PATH by running:

```bash
poetry --version
```

### 3. Install Project Dependencies

Once you have activated your virtual environment and installed Poetry, run the following command to install the project dependencies:

```bash
poetry install
```

### 4. Install PyTorch (CPU or GPU)

This project supports both CPU and GPU versions of PyTorch. Follow the appropriate instructions based on your system configuration.

#### Option 1: Install PyTorch (CPU Version)

To install the CPU version of PyTorch, use the following command:

```bash
poetry install --extras "pytorch-cpu" --source pytorch-cpu
```

#### Option 2: Install PyTorch (GPU Version)

To install the GPU version of PyTorch (CUDA 12.4), use the following command:

```bash
poetry install --extras "pytorch-gpu" --source pytorch-gpu
```

> _Note:_ For different CUDA versions, replace `cu124` with the corresponding version in the `pyproject.toml` or adjust the source URL as needed.
