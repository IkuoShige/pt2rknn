# HoST Model sim2real Deployment

This directory contains the pipeline for converting HoST PyTorch models to RKNN format for deployment on High Torque Mini Pi hardware.

## Directory Structure

```
host_sim2real/
├── models/               # Model files
│   ├── model_12000.pt   # Original HoST checkpoint
│   ├── host_model_12000.onnx  # Exported ONNX model
│   └── host_model_12000.rknn  # Converted RKNN model (to be generated)
├── scripts/             # Conversion scripts
│   ├── export_host_to_onnx.py      # PyTorch → ONNX
│   ├── convert_onnx_to_rknn.py     # ONNX → RKNN
│   └── onnx2rknn.py                # Reference script
├── configs/             # Configuration files
│   ├── host.yaml                   # Original host config
│   └── host_model_12000.yaml       # New model config
├── tests/               # Test scripts (future)
└── README.md            # This file
```

## Pipeline Overview

```
HoST Training → PyTorch (.pt) → ONNX (.onnx) → RKNN (.rknn) → Mini Pi Hardware
                    ↓               ↓               ↓
                Step 1 ✓        Step 2          Step 3
```

## Step 1: Export PyTorch to ONNX ✓ COMPLETED

Convert HoST PyTorch checkpoint to ONNX format:

```bash
cd ~/ht_ws/host_sim2real/scripts
uv run python export_host_to_onnx.py \
    --checkpoint ../models/model_12000.pt \
    --output ../models/host_model_12000.onnx
```

**Status:** ✓ Completed successfully
- Input: 258-dim observations (43 × 6 frame_stack)
- Output: 12-dim actions
- Model size: 1.17 MB
- Architecture: Actor network with [512, 256, 128] hidden layers

## Step 2: Convert ONNX to RKNN ✓ COMPLETED

Convert ONNX model to RKNN format for RK3588s NPU:

```bash
cd ~/uv_play/pt2rknn
uv run python scripts/convert_onnx_to_rknn.py \
    --input models/host_model_12000.onnx \
    --output models/host_model_12000.rknn \
    --platform rk3588s
```

**Status:** ✓ Completed successfully (x86_64 and aarch64)

**Supported platforms:**
- x86_64 Linux
- aarch64 Linux (Rockchip boards, etc.)

## Step 3: Deploy to Mini Pi Hardware

### 3.1 Copy Model to Mini Pi

```bash
# Copy RKNN model to Mini Pi policy directory
scp ~/ht_ws/host_sim2real/models/host_model_12000.rknn \
    pi@minipi:/home/pi/.../install/share/sim2real/policy/
```

### 3.2 Update Configuration

Copy the config file:

```bash
scp ~/ht_ws/host_sim2real/configs/host_model_12000.yaml \
    pi@minipi:/home/pi/.../install/share/sim2real/config/up/
```

Update `pi_rl_config.yaml` to use the new model:

```yaml
up:
  - name: "host_12000"
    path: "up/host_model_12000.yaml"
```

### 3.3 Test Deployment

```bash
# On Mini Pi
cd ~/PI_V2.0.1_beta
bash sim2real_pi.sh
```

## Model Configuration Details

### HoST Model Specifications
- **Task:** Standing-up from supine posture
- **Robot:** High Torque Mini Pi (12 DOF)
- **Observations:** 43 single-step obs × 6 frame stack = 258 total
- **Actions:** 12 joint position targets
- **Action Scale:** 1.0 (adjustable for hardware)
- **Control Frequency:** 50 Hz (RL), 200 Hz (PD)

### Observation Composition (43-dim)
From `pi_config_ground.py`:
- Base orientation (quaternion): 4
- Base angular velocity: 3
- Joint positions: 12
- Joint velocities: 12
- Last actions: 12

Total: 4 + 3 + 12 + 12 + 12 = 43

### PD Control Parameters

**Baseline (from existing config):**
```yaml
kp: [68, 31, 80, 68, 31, 31, 68, 31, 80, 68, 31, 31]
kd: [0.68, 0.68, 1.1, 0.68, 0.68, 0.68, ...]
```

**Recommended for hardware (1.5x on knee/hip):**
```yaml
kp: [102, 47, 120, 102, 47, 31, 102, 47, 120, 102, 47, 31]
kd: [1.02, 1.02, 1.65, 1.02, 1.02, 0.68, ...]
```

## Tuning Guidelines

Based on HoST README recommendations:

### 1. PD Gain Tuning
- Start with baseline gains
- Gradually increase knee and hip stiffness (1.33-1.5×)
- Monitor joint torques and motor temperatures
- Typical adjustment range: kp ± 20%, kd ± 30%

### 2. Action Scale Tuning
- Simulation: 1.0
- Hardware initial: 1.0
- Can increase to 1.2-1.3 if motion is too conservative
- Higher values = more aggressive motion

### 3. Safety Considerations
- **Always** test in a safe environment
- Have emergency stop ready
- Start with robot on soft surface
- Monitor for:
  - Excessive oscillation (reduce kd)
  - Sluggish response (increase kp)
  - Jerky motion (reduce action_scale)

### 4. Common Issues

**Robot doesn't move:**
- Increase kp gains
- Check action_scale (may be too low)
- Verify model is loaded correctly

**Robot oscillates:**
- Decrease kd gains
- Check for mechanical issues
- Verify observation data is clean

**Motion too jerky:**
- Decrease action_scale
- Increase kd for damping
- Check control frequency

## Environment Setup

### Quick Start (x86_64 / aarch64)

```bash
cd ~/uv_play/pt2rknn
uv sync
uv run python scripts/convert_onnx_to_rknn.py --help
```

This works on both x86_64 and aarch64 (e.g., Rockchip boards).

### aarch64 Support Notes

PyPIのrknn-toolkit2 2.3.2は`onnxoptimizer==0.3.8`に依存していますが、onnxoptimizerにはaarch64用プリビルドホイールがありません。

解決策として、スタブパッケージを`./dist/`に配置し、`find-links`で参照しています：

```toml
# pyproject.toml
[tool.uv]
find-links = ["./dist"]
```

スタブの再生成が必要な場合：

```bash
cd wheels/onnxoptimizer
uv build
# -> dist/onnxoptimizer-0.3.8-py3-none-any.whl
```

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.0.0 | PyTorch (CPU) |
| onnx | >=1.12.0 | ONNX format |
| onnxruntime | >=1.16.0,<1.18.0 | ONNX inference |
| numpy | <2.0.0 | rknn-toolkit2 compatibility |
| rknn-toolkit2 | >=2.3.2 | RKNN conversion |
| setuptools | * | for pkg_resources |

### Alternative: Docker (x86_64 only)

```bash
docker pull rockchip/rknn-toolkit2:latest
```

## Files Reference

### Source Files (DO NOT MODIFY)
- `/home/hightorque/PI_V2.0.1_beta/` - Original sim2real system
- `/home/hightorque/ht_ws/HoST/` - HoST training codebase

### Working Files (~/uv_play/pt2rknn/)
- `models/model_12000.pt` - HoST checkpoint (copied)
- `models/host_model_12000.onnx` - ONNX export ✓
- `models/host_model_12000.rknn` - RKNN model ✓
- `configs/host_model_12000.yaml` - Deployment config
- `scripts/export_host_to_onnx.py` - Conversion script
- `scripts/convert_onnx_to_rknn.py` - Conversion script
- `dist/onnxoptimizer-0.3.8-py3-none-any.whl` - Stub for aarch64

## Next Steps

1. ✓ Export PyTorch → ONNX (COMPLETED)
2. ✓ Convert ONNX → RKNN (COMPLETED - works on x86_64 and aarch64)
3. ⏳ Deploy to Mini Pi hardware
4. ⏳ Test and tune PD gains
5. ⏳ Validate standing-up performance

## Contact & Support

For issues related to:
- HoST model: See HoST repository
- sim2real system: See PI_V2.0.1_beta documentation
- RKNN conversion: Rockchip rknn-toolkit2 documentation

## References

- HoST Paper: https://arxiv.org/abs/2502.08378
- HoST GitHub: https://github.com/OpenRobotLab/HoST
- Mini Pi: https://www.hightorquerobotics.com/pi/
- RKNN Toolkit: https://github.com/rockchip-linux/rknn-toolkit2
