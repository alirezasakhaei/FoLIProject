"""
Debug script to figure out the correct MLP architecture.
"""

def calc_mlp_params(input_size, hidden_layers, hidden_size, num_classes):
    """Calculate total parameters for an MLP."""
    params = 0

    # First layer
    params += input_size * hidden_size + hidden_size

    # Hidden layers
    for _ in range(hidden_layers - 1):
        params += hidden_size * hidden_size + hidden_size

    # Output layer
    params += hidden_size * num_classes + num_classes

    return params

# Paper targets
target_1x512 = 1_640_458
target_3x512 = 1_901_578

print("=" * 70)
print("MLP PARAMETER ANALYSIS")
print("=" * 70)
print()

# Try different input sizes and architectures
input_sizes = {
    "28×28 (cropped)": 3 * 28 * 28,
    "32×32 (original)": 3 * 32 * 32,
}

print("Testing different architectures:")
print()

for name, input_size in input_sizes.items():
    print(f"Input size: {name} = {input_size}")
    print()

    # Try 1 hidden layer
    params_1 = calc_mlp_params(input_size, 1, 512, 10)
    diff_1 = params_1 - target_1x512
    print(f"  1 hidden layer of 512:")
    print(f"    Params: {params_1:,}")
    print(f"    Target: {target_1x512:,}")
    print(f"    Diff: {diff_1:,} ({'✓ MATCH!' if diff_1 == 0 else '✗'})")
    print()

    # Try 2 hidden layers
    params_2 = calc_mlp_params(input_size, 2, 512, 10)
    diff_2 = params_2 - target_1x512
    print(f"  2 hidden layers of 512:")
    print(f"    Params: {params_2:,}")
    print(f"    Target: {target_1x512:,}")
    print(f"    Diff: {diff_2:,} ({'✓ MATCH!' if diff_2 == 0 else '✗'})")
    print()

    # Try 3 hidden layers
    params_3 = calc_mlp_params(input_size, 3, 512, 10)
    diff_3 = params_3 - target_3x512
    print(f"  3 hidden layers of 512:")
    print(f"    Params: {params_3:,}")
    print(f"    Target: {target_3x512:,}")
    print(f"    Diff: {diff_3:,} ({'✓ MATCH!' if diff_3 == 0 else '✗'})")
    print()
    print("-" * 70)
    print()

# Work backwards to find required input size
print("=" * 70)
print("REVERSE ENGINEERING")
print("=" * 70)
print()

# For 2 hidden layers to match target_1x512
hidden_params = 512 * 512 + 512  # Second hidden layer
output_params = 512 * 10 + 10     # Output layer
remaining = target_1x512 - hidden_params - output_params

print(f"For MLP-1x512 with 2 hidden layers:")
print(f"  Hidden layer params: {hidden_params:,}")
print(f"  Output layer params: {output_params:,}")
print(f"  Remaining for first layer: {remaining:,}")

input_size_needed = (remaining - 512) / 512
print(f"  Required input size: {input_size_needed:.1f}")
print(f"  √(input_size/3) = {(input_size_needed/3)**0.5:.2f} (image dimension)")
print()

# For 3 hidden layers to match target_3x512
hidden_params = (512 * 512 + 512) * 2  # Two middle hidden layers
output_params = 512 * 10 + 10          # Output layer
remaining = target_3x512 - hidden_params - output_params

print(f"For MLP-3x512 with 3 hidden layers:")
print(f"  Hidden layers params: {hidden_params:,}")
print(f"  Output layer params: {output_params:,}")
print(f"  Remaining for first layer: {remaining:,}")

input_size_needed = (remaining - 512) / 512
print(f"  Required input size: {input_size_needed:.1f}")
print(f"  √(input_size/3) = {(input_size_needed/3)**0.5:.2f} (image dimension)")
