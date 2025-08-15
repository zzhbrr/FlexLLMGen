"""
Test script for Mixtral Cost Model.
"""

import sys
import os

# Add the parent directory to the path so we can import the cost model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimental.mixtral_cost_model import get_mixtral_config, solve_lp, solve
from flexllmgen.utils import GB

def test_basic_functionality():
    """Test basic functionality of the Mixtral cost model."""
    print("Testing Mixtral Cost Model...")
    
    # Test config creation
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    print(f"Config created successfully:")
    print(f"  Layers: {config.l}")
    print(f"  Hidden size: {config.h1}")
    print(f"  Intermediate size: {config.h2}")
    print(f"  Attention heads: {config.nh}")
    print(f"  KV heads: {config.nkv}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Experts per token: {config.num_experts_per_tok}")
    
    # Set test parameters
    config.s = 512  # prompt length
    config.n = 32   # generation length
    config.gmem = 16 * GB
    config.cmem = 200 * GB
    config.nmem = 1500 * GB
    
    print(f"\nTesting with:")
    print(f"  Prompt length: {config.s}")
    print(f"  Generation length: {config.n}")
    print(f"  GPU memory: {config.gmem / GB} GB")
    print(f"  CPU memory: {config.cmem / GB} GB")
    print(f"  NVMe memory: {config.nmem / GB} GB")
    
    # Test single policy evaluation
    print(f"\nTesting single policy evaluation...")
    bls = 48  # batch size
    gbs = 16  # gpu batch size
    
    status, policy, (throughput, tpre_tot, tgen_tot), (gpu_peak_p, gpu_peak_g) = solve_lp(
        config, bls, gbs, verbose=1
    )
    
    if status == 1:
        print(f"✓ Single policy test passed!")
        print(f"  Throughput: {throughput:.2f} token/s")
        print(f"  Prefill time: {tpre_tot:.4f} s")
        print(f"  Generation time: {tgen_tot:.4f} s")
        print(f"  GPU peak memory (prefill): {gpu_peak_p / GB:.3f} GB")
        print(f"  GPU peak memory (generation): {gpu_peak_g / GB:.3f} GB")
    else:
        print(f"✗ Single policy test failed with status: {status}")
        return False
    
    return True

def test_policy_search():
    """Test policy search functionality."""
    print(f"\nTesting policy search...")
    
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config.s = 512
    config.n = 32
    config.gmem = 16 * GB
    config.cmem = 200 * GB
    config.nmem = 1500 * GB
    
    # Create args dict for policy search
    args = {
        "gbs": None,
        "num_gb": None,
        "percent": None,
        "wg": None,
        "wc": None,
        "cg": None,
        "cc": None,
        "hg": None,
        "hc": None,
        "compress_w": False
    }
    
    try:
        best_policy, max_throughput = solve(config, solve_lp, args)
        if best_policy is not None:
            print(f"✓ Policy search test passed!")
            print(f"  Best throughput: {max_throughput:.2f} token/s")
            print(f"  Best policy: {best_policy}")
        else:
            print(f"✗ Policy search failed - no valid policy found")
            return False
    except Exception as e:
        print(f"✗ Policy search failed with error: {e}")
        return False
    
    return True

def test_fixed_policy():
    """Test with a fixed policy configuration."""
    print(f"\nTesting fixed policy configuration...")
    
    config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config.s = 512
    config.n = 32
    config.gmem = 16 * GB
    config.cmem = 200 * GB
    config.nmem = 1500 * GB
    
    # Test with a specific policy: offload weights to CPU, cache to CPU
    args = {
        "gbs": 8,
        "num_gb": 2,
        "percent": [20, 80, 0, 100, 0, 100],  # 20% weights on GPU, 80% on CPU, etc.
        "wg": None,
        "wc": None,
        "cg": None,
        "cc": None,
        "hg": None,
        "hc": None,
        "compress_w": False
    }
    
    try:
        best_policy, max_throughput = solve(config, solve_lp, args)
        if best_policy is not None:
            print(f"✓ Fixed policy test passed!")
            print(f"  Throughput: {max_throughput:.2f} token/s")
            print(f"  Policy: {best_policy}")
        else:
            print(f"✗ Fixed policy test failed - no valid policy found")
            return False
    except Exception as e:
        print(f"✗ Fixed policy test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Mixtral Cost Model Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_basic_functionality()
    all_tests_passed &= test_policy_search()
    all_tests_passed &= test_fixed_policy()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ All tests passed! Mixtral cost model is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    print("=" * 60)