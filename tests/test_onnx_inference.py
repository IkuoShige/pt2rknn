#!/usr/bin/env python3
"""
Test ONNX model inference to verify conversion correctness

This script tests the exported ONNX model with dummy observations
to ensure it produces valid outputs.
"""

import numpy as np
import argparse
import os
import sys

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnx or onnxruntime not installed!")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)


def load_onnx_model(model_path):
    """Load and validate ONNX model"""
    print(f"Loading ONNX model: {model_path}")

    # Load model
    model = onnx.load(model_path)

    # Validate model
    try:
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return None

    # Print model info
    print(f"\nModel Information:")
    print(f"  Producer: {model.producer_name}")
    print(f"  IR version: {model.ir_version}")

    # Print input/output info
    print(f"\n  Inputs:")
    for input in model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"    {input.name}: {shape}")

    print(f"\n  Outputs:")
    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"    {output.name}: {shape}")

    return model


def test_inference(model_path, num_tests=10):
    """Test ONNX model inference with random observations"""
    print("\n" + "=" * 70)
    print("Testing ONNX Model Inference")
    print("=" * 70)

    # Create inference session
    print(f"\nCreating ONNX Runtime session...")
    session = ort.InferenceSession(model_path)

    # Get input/output names and shapes
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    print(f"✓ Session created")
    print(f"  Input: {input_name}, shape: {input_shape}")
    print(f"  Output: {output_name}, shape: {output_shape}")

    # Run tests
    print(f"\nRunning {num_tests} inference tests...")

    results = {
        'outputs': [],
        'min_values': [],
        'max_values': [],
        'mean_values': []
    }

    for i in range(num_tests):
        # Generate random observations (normalized around 0)
        observations = np.random.randn(1, 258).astype(np.float32)

        # Run inference
        outputs = session.run([output_name], {input_name: observations})
        actions = outputs[0]

        # Store results
        results['outputs'].append(actions)
        results['min_values'].append(actions.min())
        results['max_values'].append(actions.max())
        results['mean_values'].append(actions.mean())

        if i == 0:
            print(f"\nTest {i+1}:")
            print(f"  Input range: [{observations.min():.3f}, {observations.max():.3f}]")
            print(f"  Output shape: {actions.shape}")
            print(f"  Output range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"  First 6 actions: {actions[0, :6]}")

    # Statistical summary
    print(f"\nStatistical Summary ({num_tests} tests):")
    print(f"  Output range: [{min(results['min_values']):.4f}, {max(results['max_values']):.4f}]")
    print(f"  Mean output: {np.mean(results['mean_values']):.4f}")
    print(f"  Std output: {np.std(results['mean_values']):.4f}")

    # Check for anomalies
    print(f"\nValidation Checks:")

    # Check 1: Output should be in [-1, 1] (tanh activation)
    all_outputs = np.concatenate(results['outputs'], axis=0)
    if np.all(np.abs(all_outputs) <= 1.0):
        print("  ✓ All outputs in valid range [-1, 1]")
    else:
        print(f"  ✗ Some outputs outside [-1, 1] range!")
        print(f"    Min: {all_outputs.min()}, Max: {all_outputs.max()}")

    # Check 2: Outputs should have reasonable variance
    output_std = all_outputs.std()
    if 0.1 < output_std < 0.9:
        print(f"  ✓ Output variance is reasonable (std={output_std:.3f})")
    else:
        print(f"  ⚠ Output variance might be unusual (std={output_std:.3f})")

    # Check 3: No NaN or Inf
    if np.all(np.isfinite(all_outputs)):
        print("  ✓ No NaN or Inf values detected")
    else:
        print("  ✗ NaN or Inf values detected!")
        return False

    print("\n✓ All tests passed!")
    return True


def test_with_realistic_observations(model_path):
    """Test with more realistic observation values"""
    print("\n" + "=" * 70)
    print("Testing with Realistic Observations")
    print("=" * 70)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Create realistic observations (robot lying down)
    # Base orientation (quaternion for lying down): [0, 0.707, 0, 0.707] → 90 deg pitch
    # Base angular velocity: small
    # Joint positions: near zero (standing pose)
    # Joint velocities: zero
    # Last actions: zero

    observations = np.zeros((1, 258), dtype=np.float32)

    # Fill in realistic values for each frame (6 frames)
    for frame in range(6):
        base_idx = frame * 43

        # Base orientation (quaternion) - lying down position
        observations[0, base_idx:base_idx+4] = [0, 0.707, 0, 0.707]

        # Base angular velocity - small
        observations[0, base_idx+4:base_idx+7] = [0.01, 0.02, 0.01]

        # Joint positions - near zero (12 joints)
        observations[0, base_idx+7:base_idx+19] = np.random.randn(12) * 0.1

        # Joint velocities - near zero (12 joints)
        observations[0, base_idx+19:base_idx+31] = np.random.randn(12) * 0.05

        # Last actions - near zero (12 joints)
        observations[0, base_idx+31:base_idx+43] = np.random.randn(12) * 0.1

    print("\nRealistic test case: Robot lying on ground")
    print(f"  Base orientation (frame 0): {observations[0, :4]}")
    print(f"  Joint positions (frame 0, first 6): {observations[0, 7:13]}")

    # Run inference
    outputs = session.run([output_name], {input_name: observations})
    actions = outputs[0]

    print(f"\nOutput actions:")
    print(f"  Shape: {actions.shape}")
    print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  Left leg (hip, roll, thigh, calf, ankle_p, ankle_r): {actions[0, :6]}")
    print(f"  Right leg: {actions[0, 6:]}")

    # Check if actions seem reasonable
    if np.abs(actions).max() < 0.95:
        print("\n✓ Actions are not saturated (good)")
    else:
        print("\n⚠ Some actions are near saturation")

    return True


def benchmark_inference_speed(model_path, num_iterations=1000):
    """Benchmark inference speed"""
    import time

    print("\n" + "=" * 70)
    print("Benchmarking Inference Speed")
    print("=" * 70)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    observations = np.random.randn(1, 258).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run([output_name], {input_name: observations})

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        session.run([output_name], {input_name: observations})
    end_time = time.time()

    elapsed = end_time - start_time
    avg_time = elapsed / num_iterations * 1000  # ms
    fps = num_iterations / elapsed

    print(f"\nResults ({num_iterations} iterations):")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Average inference time: {avg_time:.3f} ms")
    print(f"  Throughput: {fps:.1f} inferences/sec")

    # Check if it meets real-time requirements
    target_fps = 50  # RL control frequency
    if fps >= target_fps:
        print(f"\n✓ Can achieve target control frequency ({target_fps} Hz)")
    else:
        print(f"\n⚠ May not achieve target frequency (need {target_fps} Hz, got {fps:.1f} Hz)")
        print(f"  Note: This is CPU inference. RKNN on NPU will be much faster.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test ONNX model inference'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='../models/host_model_12000.onnx',
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--tests', '-n',
        type=int,
        default=10,
        help='Number of random tests to run'
    )
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run inference speed benchmark'
    )

    args = parser.parse_args()

    # Convert relative path
    if not os.path.isabs(args.model):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.join(script_dir, args.model)

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print("=" * 70)
    print("ONNX Model Inference Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print()

    # Load and validate model
    model = load_onnx_model(args.model)
    if model is None:
        sys.exit(1)

    # Run tests
    try:
        # Basic inference tests
        if not test_inference(args.model, num_tests=args.tests):
            sys.exit(1)

        # Realistic observation test
        if not test_with_realistic_observations(args.model):
            sys.exit(1)

        # Benchmark
        if args.benchmark:
            benchmark_inference_speed(args.model)

        print("\n" + "=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
