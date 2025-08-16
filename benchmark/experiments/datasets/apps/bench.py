import sys
sys.path.append("/home/zzh/llmserving/FlexLLMGen")

import argparse
from experimental.mixtral_cost_model import get_mixtral_config, solve, GB, solve_lp
from benchmark.experiments.datasets.apps.loaddata import loaddata
from flexllmgen.policy import Policy
from flexllmgen.flex_moe import run_flexllmgen
from flexllmgen.utils import str2bool

def get_policy(args) -> Policy:
    config = get_mixtral_config(args.model)

    config.s = args.prompt_len
    config.n = args.gen_len

    config.gmem = args.gpu_mem * GB
    config.cmem = args.cpu_mem * GB
    config.nmem = args.nvme_mem * GB

    best_policy, max_throughput = solve(config, solve_lp, vars(args))
    return best_policy, max_throughput

def convert_args(args, policy: Policy):
    args_input = argparse.Namespace()
    args_input.model = args.model
    args_input.path = args.path
    args_input.offload_dir = args.offload_dir
    args_input.prompt_len = args.prompt_len
    args_input.gen_len = args.gen_len
    args_input.cut_gen_len = None
    args_input.debug_mode = None
    args_input.gpu_batch_size = policy.gpu_batch_size
    args_input.num_gpu_batches = policy.num_gpu_batches
    # args_input.percent = [policy.w_gpu_percent*100, policy.w_cpu_percent*100, policy.cache_gpu_percent*100, policy.cache_cpu_percent*100, policy.act_gpu_percent*100, policy.act_cpu_percent*100]
    args_input.percent = [15, 85, 0, 100, 0, 100]
    args_input.sep_layer = True
    args_input.pin_weight = True
    args_input.cpu_cache_compute = False
    args_input.attn_sparsity = 1.0
    args_input.compress_weight = False
    args_input.compress_cache = False
    args_input.overlap = True
    args_input.log_file = args.log_file
    args_input.no_log = False
    args_input.verbose = 2
    args_input.max_sequence_length = args.max_len
    return args_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--path", type=str, default="/data1/zzh/flexgen_weights")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--log-file", type=str, default='auto')
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--min-len", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-mem", type=int, default=24)
    parser.add_argument("--cpu-mem", type=int, default=100)
    parser.add_argument("--nvme-mem", type=int, default=0)
    
    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--percent", nargs="+", type=int)
    parser.add_argument("--wg", type=int)
    parser.add_argument("--wc", type=int)
    parser.add_argument("--cg", type=int)
    parser.add_argument("--cc", type=int)
    parser.add_argument("--hg", type=int)
    parser.add_argument("--hc", type=int)
    parser.add_argument("--compress-w", action="store_true")

    parser.add_argument("--alpha-g", type=float)
    parser.add_argument("--alpha-c", type=float)
    parser.add_argument("--alpha-n", type=float)
    args = parser.parse_args()
 
    policy, max_throughput = get_policy(args)
    print(policy)

    input_args = convert_args(args, policy)

    data, sum_len = loaddata(policy.gpu_batch_size * policy.num_gpu_batches, args.min_len, args.max_len, return_sum_length=True)
    print(f"sum_len: {sum_len}")
    print(f"avg_len: {sum_len / (policy.gpu_batch_size * policy.num_gpu_batches)}")

    run_flexllmgen(input_args, data)

# python bench.py --prompt-len 685 --min-len 500 --max-len 1000 --gen-len 128 --gpu-mem 24 --cpu-mem 486