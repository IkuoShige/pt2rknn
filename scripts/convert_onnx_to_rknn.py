#!/usr/bin/env python3
"""
Convert HoST ONNX model to RKNN format for Rockchip RK3588 NPU deployment

This script converts the ONNX model exported from HoST to RKNN format
for deployment on Mini Pi hardware with RK3588s processor.

Usage:
    python convert_onnx_to_rknn.py --input <model.onnx> --output <model.rknn>
"""

import os
import sys
import argparse

try:
    from rknn.api import RKNN
except ImportError:
    print("ERROR: rknn-toolkit2 is not installed!")
    print("This script must be run on a system with rknn-toolkit2 installed.")
    print("Please install it following: https://github.com/rockchip-linux/rknn-toolkit2")
    sys.exit(1)


def convert_onnx_to_rknn(onnx_path, rknn_path, platform="rk3588s", quantize=False):
    """
    Convert ONNX model to RKNN format

    Args:
        onnx_path: Path to input ONNX model
        rknn_path: Path to output RKNN model
        platform: Target platform (default: rk3588s for Mini Pi)
        quantize: Whether to apply quantization (default: False for higher accuracy)

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    print("=" * 70)
    print("ONNX to RKNN Converter for HoST Model")
    print("=" * 70)

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        return False

    print(f"\nInput:  {onnx_path}")
    print(f"Output: {rknn_path}")
    print(f"Platform: {platform}")
    print(f"Quantization: {'Enabled' if quantize else 'Disabled'}")

    # Step 1: Create RKNN object
    print("\n[Step 1/5] Creating RKNN object...")
    rknn = RKNN(verbose=True)

    # Step 2: Config for target platform
    print(f"\n[Step 2/5] Configuring for {platform}...")
    ret = rknn.config(
        target_platform=platform,
        # For standing-up task, we want maximum accuracy
        # quantization is disabled by default
    )
    if ret != 0:
        print("ERROR: RKNN config failed!")
        return False
    print("✓ Configuration completed")

    # Step 3: Load ONNX model
    print("\n[Step 3/5] Loading ONNX model...")
    # Fix batch_size to 1 for deployment (258 = 43 obs × 6 frame_stack)
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=['observations'],
        input_size_list=[[1, 258]]
    )
    if ret != 0:
        print("ERROR: Failed to load ONNX model!")
        rknn.release()
        return False
    print("✓ ONNX model loaded successfully")

    # Step 4: Build RKNN model
    print("\n[Step 4/5] Building RKNN model...")
    print("This may take a few minutes...")
    ret = rknn.build(do_quantization=quantize)
    if ret != 0:
        print("ERROR: RKNN build failed!")
        rknn.release()
        return False
    print("✓ RKNN model built successfully")

    # Step 5: Export RKNN model
    print("\n[Step 5/5] Exporting RKNN model...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(rknn_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("ERROR: Failed to export RKNN model!")
        rknn.release()
        return False

    print(f"✓ RKNN model exported to: {rknn_path}")

    # Get file size
    file_size_kb = os.path.getsize(rknn_path) / 1024
    print(f"  File size: {file_size_kb:.2f} KB")

    # Step 6: Release resources
    print("\n[Step 6/6] Releasing resources...")
    rknn.release()
    print("✓ Done")

    print("\n" + "=" * 70)
    print("✓ Conversion completed successfully!")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert HoST ONNX model to RKNN format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (recommended)
    python convert_onnx_to_rknn.py \\
        --input ../models/host_model_12000.onnx \\
        --output ../models/host_model_12000.rknn

    # With quantization (smaller file, slightly lower accuracy)
    python convert_onnx_to_rknn.py \\
        --input ../models/host_model_12000.onnx \\
        --output ../models/host_model_12000_quant.rknn \\
        --quantize

Notes:
    - This script requires rknn-toolkit2 to be installed
    - The conversion should be run on an x86_64 Linux system or on the target device
    - For standing-up task, quantization is NOT recommended to maintain accuracy
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../models/host_model_12000.onnx',
        help='Path to input ONNX model (default: ../models/host_model_12000.onnx)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../models/host_model_12000.rknn',
        help='Path to output RKNN model (default: ../models/host_model_12000.rknn)'
    )
    parser.add_argument(
        '--platform', '-p',
        type=str,
        default='rk3588s',
        choices=['rk3588', 'rk3588s', 'rk3566', 'rk3568'],
        help='Target Rockchip platform (default: rk3588s for Mini Pi)'
    )
    parser.add_argument(
        '--quantize', '-q',
        action='store_true',
        help='Enable quantization (reduces file size but may reduce accuracy)'
    )

    args = parser.parse_args()

    # Convert relative paths to absolute (relative to current working directory)
    if not os.path.isabs(args.input):
        args.input = os.path.abspath(args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)

    # Perform conversion
    success = convert_onnx_to_rknn(
        onnx_path=args.input,
        rknn_path=args.output,
        platform=args.platform,
        quantize=args.quantize
    )

    if success:
        print("\n✓ Ready for deployment!")
        print(f"\nNext steps:")
        print(f"  1. Copy {args.output} to Mini Pi")
        print(f"  2. Place it in /home/pi/.../install/share/sim2real/policy/")
        print(f"  3. Update config file to use this model")
        sys.exit(0)
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
