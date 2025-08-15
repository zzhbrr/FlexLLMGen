"""
Example usage of Mixtral Cost Model.

This script demonstrates how to use the Mixtral cost model to find optimal policies
for running Mixtral 8x7B models with limited GPU memory.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimental.mixtral_cost_model import get_mixtral_config, solve_lp, solve
from flexllmgen.utils import GB

def example_find_best_policy():
    """Example: Find the best policy for given hardware constraints."""
    print("Example 1: Finding the best policy for Mixtral 8x7B")
    print("=" * 60)
    
    # Get model configuration
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    # Set workload parameters
    config.s = 512  # prompt length
    config.n = 32   # generation length
    
    # Set hardware constraints
    config.gmem = 16 * GB    # 16GB GPU memory
    config.cmem = 200 * GB   # 200GB CPU memory  
    config.nmem = 1500 * GB  # 1500GB NVMe storage
    
    print(f"Hardware constraints:")
    print(f"  GPU memory: {config.gmem / GB} GB")
    print(f"  CPU memory: {config.cmem / GB} GB")
    print(f"  NVMe storage: {config.nmem / GB} GB")
    print(f"Workload:")
    print(f"  Prompt length: {config.s}")
    print(f"  Generation length: {config.n}")
    print()
    
    # Search for the best policy
    args = {
        "gbs": None,
        "num_gb": None,
        "percent": None,
        "wg": None, "wc": None, "cg": None, "cc": None, "hg": None, "hc": None,
        "compress_w": False
    }
    
    best_policy, max_throughput = solve(config, solve_lp, args)
    
    if best_policy:
        print(f"Best policy found!")
        print(f"  Maximum throughput: {max_throughput:.2f} token/s")
        print(f"  GPU batch size: {best_policy.gpu_batch_size}")
        print(f"  Number of GPU batches: {best_policy.num_gpu_batches}")
        print(f"  Weight distribution: {best_policy.w_gpu_percent:.1%} GPU, {best_policy.w_cpu_percent:.1%} CPU")
        print(f"  Cache distribution: {best_policy.cache_gpu_percent:.1%} GPU, {best_policy.cache_cpu_percent:.1%} CPU")
        print(f"  Activation distribution: {best_policy.act_gpu_percent:.1%} GPU, {best_policy.act_cpu_percent:.1%} CPU")
    else:
        print("No valid policy found for the given constraints.")
    print()

def example_evaluate_specific_policy():
    """Example: Evaluate a specific policy configuration."""
    print("Example 2: Evaluating a specific policy")
    print("=" * 60)
    
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config.s = 512
    config.n = 32
    config.gmem = 16 * GB
    config.cmem = 200 * GB
    config.nmem = 1500 * GB
    
    # Evaluate a specific policy: 20% weights on GPU, 80% on CPU
    # All cache and activations on CPU
    bls = 32  # total batch size
    gbs = 8   # GPU batch size
    
    print(f"Evaluating policy:")
    print(f"  Total batch size: {bls}")
    print(f"  GPU batch size: {gbs}")
    print(f"  Weight distribution: 20% GPU, 80% CPU")
    print(f"  Cache distribution: 100% CPU")
    print(f"  Activation distribution: 100% CPU")
    print()
    
    # Use debug mode to fix the policy
    args = {
        "gbs": gbs,
        "num_gb": bls // gbs,
        "percent": [20, 80, 0, 100, 0, 100],
        "wg": None, "wc": None, "cg": None, "cc": None, "hg": None, "hc": None,
        "compress_w": False
    }
    
    best_policy, throughput = solve(config, solve_lp, args)
    
    if best_policy:
        print(f"Policy evaluation results:")
        print(f"  Throughput: {throughput:.2f} token/s")
        print(f"  Policy details: {best_policy}")
    else:
        print("Policy evaluation failed.")
    print()

def example_memory_constrained():
    """Example: Find policy for very limited GPU memory."""
    print("Example 3: Policy for limited GPU memory (8GB)")
    print("=" * 60)
    
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config.s = 256  # shorter prompts
    config.n = 16   # shorter generation
    config.gmem = 8 * GB     # Only 8GB GPU memory
    config.cmem = 100 * GB   # 100GB CPU memory
    config.nmem = 500 * GB   # 500GB NVMe storage
    
    print(f"Constrained hardware:")
    print(f"  GPU memory: {config.gmem / GB} GB")
    print(f"  CPU memory: {config.cmem / GB} GB")
    print(f"  NVMe storage: {config.nmem / GB} GB")
    print(f"Workload:")
    print(f"  Prompt length: {config.s}")
    print(f"  Generation length: {config.n}")
    print()
    
    args = {
        "gbs": None,
        "num_gb": None,
        "percent": None,
        "wg": None, "wc": None, "cg": None, "cc": None, "hg": None, "hc": None,
        "compress_w": False
    }
    
    best_policy, max_throughput = solve(config, solve_lp, args)
    
    if best_policy:
        print(f"Policy for constrained memory:")
        print(f"  Maximum throughput: {max_throughput:.2f} token/s")
        print(f"  GPU batch size: {best_policy.gpu_batch_size}")
        print(f"  Number of GPU batches: {best_policy.num_gpu_batches}")
        print(f"  Weight distribution: {best_policy.w_gpu_percent:.1%} GPU, {best_policy.w_cpu_percent:.1%} CPU")
        print(f"  Cache distribution: {best_policy.cache_gpu_percent:.1%} GPU, {best_policy.cache_cpu_percent:.1%} CPU")
    else:
        print("No valid policy found for the constrained memory.")
    print()

if __name__ == "__main__":
    print("Mixtral Cost Model Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_find_best_policy()
    example_evaluate_specific_policy() 
    example_memory_constrained()
    
    print("Examples completed!")