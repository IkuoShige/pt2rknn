# pt2rknn

PyTorch → ONNX → RKNN conversion tool. Supports both x86_64 and aarch64.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/IkuoShige/pt2rknn.git
cd pt2rknn
uv sync
uv run python scripts/convert_onnx_to_rknn.py --help
```

## Usage

### PyTorch → ONNX

```bash
uv run python scripts/export_host_to_onnx.py \
    --checkpoint models/model_12000.pt \
    --output models/host_model_12000.onnx
```

### ONNX → RKNN

```bash
uv run python scripts/convert_onnx_to_rknn.py \
    --input models/host_model_12000.onnx \
    --output models/host_model_12000.rknn \
    --platform rk3588s
```

## Deploy to Mini Pi

### Copy Model

```bash
scp models/host_model_12000.rknn \
    hightorque@<ip-address>:/home/hightorque/PI_V2.0.1_beta/install/share/sim2real/policy/
```

### Copy Config: On Mini Pi

```bash
cp /home/hightorque/PI_V2.0.1_beta/install/share/sim2real/config/up/host.yaml \
   /home/hightorque/PI_V2.0.1_beta/install/share/sim2real/config/up/host_model_12000.yaml
```

Update `pi_rl_config.yaml` to use the new model:

```yaml
up:
  - name: "host_12000"
    path: "up/host_model_12000.yaml"
```

Update 'host_model_12000.yaml' to use new weight:
```yaml
policy_name: "host_model_12000.rknn"
```

### Test

```bash
# On Mini Pi
cd ~/PI_V2.0.1_beta
bash sim2real_pi.sh
```

## Directory Structure

```
pt2rknn/
├── models/                 # Model files
├── scripts/                # Conversion scripts
├── configs/                # Deployment configs
├── dist/                   # onnxoptimizer stub (for aarch64)
├── wheels/onnxoptimizer/   # Stub source
└── HoST/                   # submodule (optional)
```

## aarch64 Support

rknn-toolkit2 depends on `onnxoptimizer==0.3.8`, but PyPI has no aarch64 wheel.

Solution: Place a stub package in `dist/` and reference via `find-links`.

```toml
# pyproject.toml
[tool.uv]
find-links = ["./dist"]
```

### Rebuilding the stub

```bash
mkdir -p dist  # Required for find-links
uv build wheels/onnxoptimizer --out-dir dist
```

## Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.0.0 | CPU |
| onnx | >=1.12.0 | |
| onnxruntime | >=1.16.0,<1.18.0 | |
| numpy | <2.0.0 | rknn-toolkit2 compat |
| rknn-toolkit2 | >=2.3.2 | |
| setuptools | * | for pkg_resources |

## References

- [HoST](https://github.com/HighTorque-Robotics/HoST.git)
- [rknn-toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
