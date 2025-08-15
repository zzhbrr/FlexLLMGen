"""
Usage:
python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
import torch
import json
from transformers import AutoTokenizer

from flexllmgen.compression import CompressionConfig
from flexllmgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.timer import timers
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()

from flexllmgen.models.mixtral import Mixtral, DUMMY_WEIGHT

from flexllmgen.models.mixtral import MixtralConfig
from flexllmgen.policy import Policy

import os
os.environ["HF_HOME"] = "/data1/zzh/huggingface"


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    with open('/home/zzh/llmserving/FlexLLMGen/flexllmgen/datasets/mtbench/question.jsonl', 'r') as file:
        prompts = []
        for i, line in enumerate(file):
            if i >= num_prompts:
                break
            item = json.loads(line)
            prompts.append(item["turns"][0])
    print(prompts)
    # prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len, truncation=True).input_ids
    return input_ids

def calc_model_cache_hidden_size(config: MixtralConfig, model_name, batch_size, seq_len):
    # model size
    if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        model_size = 86.99 # GB
        expert_size = 84.00 # GB
    else:
        assert False, "Model not supported yet"
    # cache size
    cache_size = 2 * batch_size * seq_len * config.num_hidden_layers * config.head_dim * config.num_key_value_heads * 2 # Bytes
    # hidden size
    hidden_size = batch_size * seq_len * config.hidden_size * 2 # Bytes

    return model_size * GB, cache_size, hidden_size

def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    if args.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        assert False, "Model not supported yet"
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(2, num_prompts, tokenizer)
    print("warmup_inputs:", warmup_inputs)
    # with open('/home/zzh/llmserving/FlexLLMGen/flexllmgen/tests/output2.txt', 'w') as f:
    #     f.write(str(warmup_inputs))
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    assert args.compress_weight == False, "Not implemented"
    assert args.compress_cache == False, "Not implemented"

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    mixtral_config = MixtralConfig.from_pretrained(args.model)
    mixtral_config.torch_dtype = np.float16
    # for debugging
    # mixtral_config.head_dim = 16
    # mixtral_config.intermediate_size = 512
    # mixtral_config.num_attention_heads = 8
    # mixtral_config.num_key_value_heads = 4
    mixtral_config.num_hidden_layers = 32
    # mixtral_config.hidden_size = 128
    model_size, cache_size, hidden_size = calc_model_cache_hidden_size(mixtral_config, args.model, num_prompts, prompt_len + gen_len)
    print(f"model size: {model_size/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    
    model = Mixtral(mixtral_config, args.model, env, args.path, policy, tokenizer.pad_token_id)

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)
        
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        model_size, cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/data1/zzh/flexgen_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexllmgen(args)

# python flex_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --percent 0 100 0 100 0 100