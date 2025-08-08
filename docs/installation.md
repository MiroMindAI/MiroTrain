# MiroTrain Installation Guide

### Docker Installation

For the fastest setup, we provide a pre-built Docker image with all dependencies pre-installed:

```bash
# Pull the Docker image
docker pull miromind/mirotrain:base-cu121-torch2.5.1-fa2.7.4

# Run the container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  miromind/mirotrain:base-cu121-torch2.5.1-fa2.7.4
```

**Advantages:**
- No need to install dependencies manually
- Consistent environment across different systems
- Includes all required packages and optimizations
- Ready to use immediately

### Manual Installation

#### Step 1: Create Python Environment

We recommend using conda to create a clean Python 3.10 environment:

```bash
# Create conda environment
conda create --name mirotrain-env python=3.10 -y
conda activate mirotrain-env
```

#### Step 2: Install PyTorch

Install PyTorch based on your CUDA version. For CUDA 12.1:

```bash
# Install PyTorch with CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

For other PyTorch and CUDA versions, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

#### Step 3: Install MiroTrain

Clone the repository and install MiroTrain:

```bash
# Clone the repository
git clone https://github.com/MiroMindAsia/mirotrain.git
cd mirotrain

# Install TorchTune first
pip install ./torchtune

# Install MiroTrain
pip install .
```
### Verification

To verify your installation, run:

```bash
# Check if MiroTrain is installed correctly
python -c "import mirotrain; print('MiroTrain installed successfully!')"

# Check if TorchTune is available
python -c "import torchtune; print('TorchTune available!')"
```

Verify that the command-line tools are working:

```bash
# Check TorchTune CLI
tune --help
```