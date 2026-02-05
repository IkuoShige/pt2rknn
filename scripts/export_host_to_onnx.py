#!/usr/bin/env python3
"""
Export HoST PyTorch model to ONNX format for sim2real deployment

This script extracts the actor network from a HoST checkpoint and exports it to ONNX format.
The exported model takes observations (258-dim) and outputs actions (12-dim).

Usage:
    python export_host_to_onnx.py --checkpoint <path_to_model.pt> --output <output.onnx>
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# Add HoST paths to enable imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "HoST")
sys.path.insert(0, os.path.join(HOST_DIR, "rsl_rl"))

from rsl_rl.modules.actor_critic import ActorCritic

class ActorWrapper(nn.Module):
    """Wrapper to extract only the actor network for inference"""
    def __init__(self, actor_critic_model):
        super(ActorWrapper, self).__init__()
        self.actor = actor_critic_model.actor

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape (batch_size, num_observations)
                         For HoST Pi: (batch_size, 258) = (batch, 43 * 6)

        Returns:
            actions: torch.Tensor of shape (batch_size, num_actions)
                    For HoST Pi: (batch_size, 12)
        """
        return self.actor(observations)


def load_checkpoint(checkpoint_path):
    """Load HoST checkpoint and extract model info"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Print checkpoint structure
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")

    return checkpoint


def create_actor_from_checkpoint(checkpoint, num_observations=258, num_actions=12):
    """
    Create ActorCritic model and load weights from checkpoint

    Args:
        checkpoint: Loaded checkpoint dictionary
        num_observations: Number of observation dimensions (default: 258 for Pi)
        num_actions: Number of action dimensions (default: 12 for Pi)

    Returns:
        ActorWrapper: Wrapped actor network
    """
    # HoST Pi configuration (from pi_config_ground.py PiCfgPPO)
    actor_hidden_dims = [512, 256, 128]
    num_critics = 4  # Pi uses 4 critics

    print(f"\nCreating ActorCritic model:")
    print(f"  num_observations: {num_observations}")
    print(f"  num_actions: {num_actions}")
    print(f"  actor_hidden_dims: {actor_hidden_dims}")

    # Create ActorCritic model
    # Critic hidden dims from PiCfgPPO
    critic_hidden_dims = [512, 256]

    actor_critic = ActorCritic(
        num_actor_obs=num_observations,
        num_critic_obs=num_observations,  # Not used in export
        num_actions=num_actions,
        num_critics=num_critics,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation='elu',
        init_noise_std=0.8  # From PiCfgPPO
    )

    # Load state dict
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()

    print("✓ Model loaded successfully")

    # Wrap to extract only actor
    actor_wrapper = ActorWrapper(actor_critic)

    return actor_wrapper


def export_to_onnx(model, output_path, num_observations=258, num_actions=12):
    """
    Export actor network to ONNX format

    Args:
        model: ActorWrapper model
        output_path: Path to save ONNX file
        num_observations: Input dimension
        num_actions: Output dimension
    """
    print(f"\nExporting to ONNX: {output_path}")

    # Create dummy input for tracing
    dummy_input = torch.randn(1, num_observations)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['actions'],
        dynamic_axes={
            'observations': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )

    print("✓ ONNX export completed")

    # Verify export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Print model info
    print(f"\nONNX Model Info:")
    print(f"  Input: {onnx_model.graph.input[0].name}, shape: [batch_size, {num_observations}]")
    print(f"  Output: {onnx_model.graph.output[0].name}, shape: [batch_size, {num_actions}]")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def test_inference(model, num_observations=258):
    """Test inference with dummy data to verify model works"""
    print("\nTesting inference with dummy data...")

    model.eval()
    with torch.no_grad():
        dummy_obs = torch.randn(1, num_observations)
        output = model(dummy_obs)

        print(f"  Input shape: {dummy_obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output sample: {output[0, :3].numpy()}")

    print("✓ Inference test passed")


def main():
    parser = argparse.ArgumentParser(description='Export HoST model to ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default='../models/model_12000.pt',
                        help='Path to HoST checkpoint (.pt file)')
    parser.add_argument('--output', type=str,
                        default='../models/host_model_12000.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--num-obs', type=int, default=258,
                        help='Number of observations (default: 258 for Pi with frame_stack=6)')
    parser.add_argument('--num-actions', type=int, default=12,
                        help='Number of actions (default: 12 for Pi)')

    args = parser.parse_args()

    # Convert relative paths to absolute (relative to current working directory)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)

    print("=" * 70)
    print("HoST Model to ONNX Exporter")
    print("=" * 70)

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)

    # Create actor model
    actor_model = create_actor_from_checkpoint(
        checkpoint,
        num_observations=args.num_obs,
        num_actions=args.num_actions
    )

    # Test inference
    test_inference(actor_model, num_observations=args.num_obs)

    # Export to ONNX
    export_to_onnx(
        actor_model,
        args.output,
        num_observations=args.num_obs,
        num_actions=args.num_actions
    )

    print("\n" + "=" * 70)
    print("✓ Export completed successfully!")
    print(f"  ONNX model saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
