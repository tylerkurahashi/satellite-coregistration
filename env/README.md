# Satellite Coregistration Development Environment

This directory contains Docker configuration for the satellite coregistration project, optimized for SpaceNet9 and similar geospatial ML tasks.

## Quick Start

### Using Docker Compose

1. Build and start the container:
```bash
docker-compose up -d --build
```

2. Access the container via SSH:
```bash
ssh -p 22222 root@localhost
```

3. Stop and remove the container:
```bash
docker-compose down
```

## Environment Configuration

### Base Image
- **NVIDIA CUDA**: 11.8.0 with cuDNN 8 (GPU acceleration support)
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10 (default system Python)

### Development Tools
- **Node.js**: 24.1.0 (installed from official Node.js binaries)
- **npm**: Latest version compatible with Node.js 24
- **GitHub CLI (gh)**: Latest version for repository management
- **Claude Code**: Anthropic's AI coding assistant CLI
- **SSH Server**: OpenSSH for remote development

### Python Dependencies (ML/DL Stack)
- **Deep Learning Frameworks**:
  - PyTorch 2.0.1 with CUDA 11.8 support
  - torchvision 0.15.2
  - timm 0.9.2 (PyTorch Image Models)
  - einops 0.6.1 (tensor operations)
  
- **Geospatial Libraries**:
  - GDAL (system library + Python bindings)
  - rasterio 1.3.8 (raster I/O)
  - geopandas 0.13.2 (geospatial dataframes)
  - shapely 2.0.1 (geometric operations)
  - pyproj 3.6.0 (coordinate transformations)
  
- **Computer Vision**:
  - OpenCV 4.8.0
  - scikit-image 0.21.0
  - albumentations 1.3.1 (image augmentations)
  - kornia 0.7.0 (differentiable CV)
  - Pillow 10.0.0
  
- **ML/Data Science**:
  - numpy 1.24.3
  - pandas 2.0.3
  - scikit-learn 1.3.0
  - scipy 1.11.1
  - matplotlib 3.7.2
  - seaborn 0.12.2
  
- **Experiment Tracking**:
  - wandb 0.15.8
  - mlflow 2.5.0
  - tensorboard 2.13.0
  
- **Development Tools**:
  - pytest 7.4.0
  - black 23.7.0 (code formatter)
  - flake8 6.1.0 (linter)
  - isort 5.12.0 (import sorter)

### Container Configuration

#### Volumes
- `../:/workspace` - Maps the repository root to `/workspace` in the container

#### Environment Variables
- `CUDA_HOME=/usr/local/cuda` - CUDA installation path
- `PYTHONPATH=/workspace` - Python module search path
- `GDAL_DATA=/usr/share/gdal` - GDAL data files
- `PROJ_LIB=/usr/share/proj` - PROJ library data
- `NVIDIA_VISIBLE_DEVICES=all` - GPU visibility
- `CUDA_VISIBLE_DEVICES=0` - Default GPU device

#### Exposed Ports
- **22222**: SSH server for remote development

### SSH Access
- **Username**: root
- **Password**: Edit Password and uncomment in Dockerfile
- **Port**: 22222

### Working Directory
The default working directory is `/workspace`, which maps to your repository root.

## Local Development (Without Docker)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install GDAL (system-specific):
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Windows
# Use OSGeo4W or conda
```

## GPU Support

### Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5+
- NVIDIA drivers (version 450.80.02 or later)
- Docker with NVIDIA Container Toolkit (nvidia-docker2)

### Enabling GPU in Docker Compose
The GPU support is currently commented out in `docker-compose.yml`. To enable:

1. Uncomment the GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

2. Rebuild and restart the container:
```bash
docker-compose down
docker-compose up -d --build
```

### Testing GPU Availability
```bash
# Inside the container
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check NVIDIA drivers
nvidia-smi
```

## Troubleshooting

### GDAL Import Error
If you encounter GDAL import errors, ensure:
1. GDAL system libraries are installed
2. Python GDAL version matches system version
3. Set environment variables:
```bash
export GDAL_DATA=/usr/share/gdal
export PROJ_LIB=/usr/share/proj
```

### Out of Memory
For large images, adjust:
- Batch size in training
- Tile size for inference
- Use gradient accumulation
- Enable mixed precision training

### Permission Denied
If running as non-root user, ensure proper permissions:
```bash
sudo chown -R $USER:$USER ../
```

### SSH Connection Issues
If you cannot connect via SSH:
1. Ensure the container is running: `docker-compose ps`
2. Check if port 22222 is available: `lsof -i :22222`
3. Verify SSH service inside container: `docker exec sat-coregistration-dev service ssh status`

### Node.js/npm Issues
The container includes Node.js 24.1.0. If you encounter issues:
```bash
# Check versions
node --version  # Should show v24.1.0
npm --version

# Reinstall global packages if needed
npm install -g @anthropic-ai/claude-code
```

## Development Workflow

### Using Claude Code
Claude Code is pre-installed in the container:
```bash
# Inside the container
claude --help

# Start coding with Claude
cd /workspace
claude
```
