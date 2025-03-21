# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, Optional, Set, Tuple, Union, List

import torch
from torch import nn
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding

import numpy as np
from tqdm import tqdm

from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from flexllmgen.policy import Policy
from flexllmgen.timer import timers
from flexllmgen.pytorch_backend import DeviceType, general_copy

import os
import glob
import shutil
import safetensors

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

def download_mixtral_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        hf_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # folder = snapshot_download(hf_model_name)
    folder = "/data1/zzh/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1"
    param_files = glob.glob(os.path.join(folder, "*.safetensors"))

    # print("folder: ", folder)
    # print("param_files: ", param_files)

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    # print("path: ", path)

    for param_file in tqdm(param_files, desc="Convert format"):
        state = safetensors.torch.load_file(param_file)
        for name, param in tqdm(state.items(), leave=False):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.to(torch.float16).cpu().detach().numpy())

def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config: MixtralConfig, env: ExecutionEnv, policy: Policy, pad_token_id: int):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

        self.rotary_emb = MixtralRotaryEmbedding(config=config, device=torch.device("cuda:0"))
        self.pad_token_id = pad_token_id
        
    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.hidden_size,
            self.config.max_position_embeddings, self.config.torch_dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "model.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights[0])

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store(w_token.smart_copy(dst))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 3
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            w_token, donate[2] = weight_read_buf.pop()
        else:
            w_token, _ = weight_read_buf.val
        
        h, pe = self.compute.mixtral_input_embed(h, mask, w_token, self.rotary_emb, self.pad_token_id, donate, self.config)
        # with open("/home/zzh/llmserving/FlexLLMGen/flexllmgen/tests/output2_inputembed.txt", "w") as f:
        #     torch.set_printoptions(profile="full")
        #     f.write(str(h.data))
        hidden.val = (h, pe) # XXX: 这里额外存了一个pe，后面的层的推理需要改改


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size,
            self.config.torch_dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "model.norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.torch_dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        (h, pe), donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.mixtral_output_embed(h, w_ln, w_token, donate,
            self.task.do_sample, self.task.temperature, self.config.rms_norm_eps)
        hidden.val = h


class MixtralAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.torch_dtype)
        h_kv = self.config.num_key_value_heads * self.config.head_dim
        weight_specs = [
            # w_q
            ((h, h), dtype, path + f"model.layers.{self.layer_id}.self_attn.q_proj.weight"),
            # w_k
            ((h_kv, h), dtype, path + f"model.layers.{self.layer_id}.self_attn.k_proj.weight"),
            # w_v
            ((h_kv, h), dtype, path + f"model.layers.{self.layer_id}.self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + f"model.layers.{self.layer_id}.self_attn.o_proj.weight"),
            # w_ln
            ((h,), dtype, path + f"model.layers.{self.layer_id}.input_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, w_k, w_v, w_out, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_out.smart_copy(dst1),
                w_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, k_home.shape[0]),  # (b, n_head, s, head_dim)
                       slice(0, k_home.shape[1]),
                       slice(0, self.task.prompt_len + i))
            # indices = (slice(0, self.task.prompt_len + i),
            #            slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, k_buf.shape[0]),  # (b, n_head, s, head_dim)
                       slice(0, k_buf.shape[1]),
                       slice(0, self.task.prompt_len + i - 1))
            # indices = (slice(0, self.task.prompt_len + i - 1),
            #            slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]),
                       slice(0, k_new.shape[2]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]),
                       slice(pos - k_new.shape[2], pos))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.torch_dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 14
        (h, pe), donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[3]), (w_v, donate[4]), (w_out, donate[5]),
             (w_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _), (w_v, _), (w_out, _),
             (w_ln, _)) = weight_read_buf.val
        
        if i == 0: # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.MixtralAttention_Prefill(h, pe, mask, w_q, w_k, w_v, w_out, w_ln, donate,
                self.policy.compress_cache, self.policy.comp_cache_config, self.config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else: # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.MixtralAttention_Decode(h, pe, mask, w_q, w_k, w_v, w_out, w_ln, donate,
                self.policy.attn_sparsity, self.policy.compress_cache, self.policy.comp_cache_config, self.config, k_cache, v_cache)
            cache_write_buf.store((new_k_cache, new_v_cache))

       # if self.layer_id == 0:
        #     with open("/home/zzh/llmserving/FlexLLMGen/flexllmgen/tests/output2_attention.txt", "w") as f:
        #         torch.set_printoptions(profile="full")
        #         torch.set_printoptions(precision=5,sci_mode=False)
        #         f.write(str(h.data))

        hidden.val = (h, pe)

class MixtralGate:
    def __init__(self, config: MixtralConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.num_experts = config.num_local_experts

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.torch_dtype)
        weight_specs = [
            ((self.num_experts, h), dtype, path + f"model.layers.{self.layer_id}.block_sparse_moe.gate.weight"),
            ((h,), dtype, path + f"model.layers.{self.layer_id}.post_attention_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self, weight_home, weight_read_buf, k):
        w_gate, w_ln = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_gate.smart_copy(dst), w_ln.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.torch_dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 1
        if k == self.policy.num_gpu_batches - 1:
            (w_gate, _), (w_ln, _) = weight_read_buf.pop()
        else:
            (w_gate, _), (w_ln, _) = weight_read_buf.val
        (h, pe), donate[0] = hidden.val, False
        pre_norm = h
        after_norm, routing_logits = self.compute.mixtral_gate(h, w_gate, w_ln, donate, self.config)

        # if self.layer_id == 0:
        #     with open("/home/zzh/llmserving/FlexLLMGen/flexllmgen/tests/output2_routinglogits.txt", "w") as f:
        #         torch.set_printoptions(profile="full")
        #         torch.set_printoptions(precision=5,sci_mode=False)
        #         f.write(str(routing_logits.data))
        
        hidden.val = (pre_norm, after_norm, routing_logits, pe)


class MixtralSparseMLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        
        self.num_experts = config.num_local_experts
        self.topk = config.num_experts_per_tok

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype, inter_h = (self.config.hidden_size, self.config.torch_dtype, self.config.intermediate_size)
        weight_specs = []
        for i in range(self.num_experts):
            tmp_path = os.path.join(path, f"model.layers.{self.layer_id}.block_sparse_moe.experts.{i}.")
            weight_specs.append(((inter_h, h), dtype, tmp_path + "w1.weight"))
            weight_specs.append(((inter_h, h), dtype, tmp_path + "w3.weight"))
            weight_specs.append(((h, inter_h), dtype, tmp_path + "w2.weight"))
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        tmp = weight_home.val
        w1, w2, w3 = [], [], []
        cnt = 0
        for i in range(self.num_experts):
            w1.append(tmp[cnt])
            w2.append(tmp[cnt + 1])
            w3.append(tmp[cnt + 2])
            cnt += 3
        if k == 0:
            dst2 = self.compute
            res = []
            for i in range(self.num_experts):
                res.append(w1[i].smart_copy(dst2))
                res.append(w2[i].smart_copy(dst2))
                res.append(w3[i].smart_copy(dst2))
            weight_read_buf.store(tuple(res))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.torch_dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        (pre_norm, h, routing_logits, pe), donate[0] = hidden.val, True # pre_norm is for residual connection
        w1, w2, w3 = [], [], []

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            tmp = weight_read_buf.pop()
            cnt = 0
            for i in range(self.num_experts):
                # 忽略donate
                w1.append(tmp[cnt][0]) 
                w2.append(tmp[cnt + 1][0])
                w3.append(tmp[cnt + 2][0])
                cnt += 3
        else:
            tmp = weight_read_buf.val
            cnt = 0
            for i in range(self.num_experts):
                # 忽略donate
                w1.append(tmp[cnt][0])
                w2.append(tmp[cnt + 1][0])
                w3.append(tmp[cnt + 2][0])
                cnt += 3
        h = self.compute.mixtral_moe(h, routing_logits, pre_norm, self.topk, self.num_experts, w1, w2, w3, donate)
        # if self.layer_id == 31:
        #     with open("/home/zzh/llmserving/FlexLLMGen/flexllmgen/tests/output2_moe.txt", "w") as f:
        #         torch.set_printoptions(profile="full")
        #         torch.set_printoptions(precision=5,sci_mode=False)
        #         f.write(str(h.data))
        hidden.val = (h, pe)


# class MLP:
#     def __init__(self, config, env, policy, layer_id):
#         self.config = config
#         self.env = env
#         self.layer_id = layer_id
#         self.policy = policy
#         self.compute = self.env.gpu
#         self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
#             else self.compute)

#         self.task = None

#     def set_task(self, task):
#         self.task = task

#     def init_weight(self, weight_home, path):
#         h, dtype = (self.config.input_dim, self.config.dtype)
#         path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
#         weight_specs = [
#             # wi
#             ((4 * h, h), dtype, path + "fc1.weight"),
#             # bi
#             ((4 * h,), dtype, path + "fc1.bias"),
#             # wo
#             ((h, 4 * h), dtype, path + "fc2.weight"),
#             # bo
#             ((h,), dtype, path + "fc2.bias"),
#             # w_ln
#             ((h,), dtype, path + "final_layer_norm.weight"),
#             # b_ln
#             ((h,), dtype, path + "final_layer_norm.bias"),
#         ]
#         weights = init_weight_list(weight_specs, self.policy, self.env)
#         weight_home.store(weights)

#     def load_weight(self, weight_home, weight_read_buf, k):
#         wi, bi, wo, bo, w_ln, b_ln = weight_home.val
#         if k == 0:
#             dst1 = self.weight_load_dst
#             dst2 = self.compute
#             weight_read_buf.store((
#                 wi.smart_copy(dst1), bi.smart_copy(dst2),
#                 wo.smart_copy(dst1), bo.smart_copy(dst2),
#                 w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

#     def init_cache_one_gpu_batch(self, cache_home):
#         pass  # do nothing

#     def load_cache(self, cache_home, cache_read_buf, i):
#         pass  # do nothing

#     def store_cache(self, cache_home, cache_write_buf, i):
#         pass  # do nothing

#     def input_act_shape_and_dtype(self, batch_size, seq_len):
#         return (batch_size, seq_len, self.config.input_dim), self.config.dtype

#     def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
#                 cache_write_buf, i, k):
#         donate = [False] * 7
#         h, donate[0] = hidden.val, True

#         if k == self.policy.num_gpu_batches - 1:
#             # Clear the weight_read_buf if it is the last gpu batch
#             ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
#              (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
#         else:
#             ((wi, _), (bi, _), (wo, _), (bo, _),
#              (w_ln, _), (b_ln, _)) = weight_read_buf.val

#         h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
#         hidden.val = h


# class TransformerLayer:
#     def __init__(self, config, env, policy, i):
#         self.attention = SelfAttention(config, env, policy, i)
#         self.mlp = MLP(config, env, policy, i)
#         self.policy = policy
#         self.compute = self.attention.compute

#     def set_task(self, task):
#         self.attention.set_task(task)
#         self.mlp.set_task(task)

#     def init_weight(self, weight_home, path):
#         home1, home2 = ValueHolder(), ValueHolder()
#         self.attention.init_weight(home1, path)
#         self.mlp.init_weight(home2, path)
#         weight_home.store((home1, home2))

#     def load_weight(self, weight_home, weight_read_buf, k):
#         read_buf1, read_buf2 = ValueHolder(), ValueHolder()
#         home1, home2 = weight_home.val
#         self.attention.load_weight(home1, read_buf1, k)
#         self.mlp.load_weight(home2, read_buf2, k)
#         if k == 0:
#             weight_read_buf.store((read_buf1, read_buf2))

#     def init_cache_one_gpu_batch(self, cache_home):
#         self.attention.init_cache_one_gpu_batch(cache_home)

#     def load_cache(self, cache_home, cache_read_buf, i):
#         self.attention.load_cache(cache_home, cache_read_buf, i)

#     def store_cache(self, cache_home, cache_write_buf, i):
#         self.attention.store_cache(cache_home, cache_write_buf, i)

#     def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
#                 cache_write_buf, i, k):
#         if k == self.policy.num_gpu_batches - 1:
#             read_buf1, read_buf2 = weight_read_buf.pop()
#         else:
#             read_buf1, read_buf2 = weight_read_buf.val

#         self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
#                                cache_write_buf, i, k)
#         self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)


class Mixtral:
    def __init__(self,
                 config: MixtralConfig,
                 model_name: str, 
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy, 
                 pad_token_id: int):
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.model_name = model_name
        self.pad_token_id = pad_token_id

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy, self.pad_token_id))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(MixtralAttention(self.config, self.env, self.policy, i))
                layers.append(MixtralGate(self.config, self.env, self.policy, i))
                layers.append(MixtralSparseMLP(self.config, self.env, self.policy, i))
            else:
                assert(False)
                # layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.model_name}-np")))
        check_path = os.path.join(expanded_path, "model.embed_tokens.weight")
        # print(expanded_path)
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_mixtral_weights(self.model_name, expanded_path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path + "/")

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            val = self.weight_home[j].pop()
            if isinstance(val, tuple) or isinstance(val, list):
                for x in val:
                    if isinstance(x, ValueHolder):
                        for y in x.pop():
                            y.delete()
                    else:
                        x.delete()
            else:
                val.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            # XXX: hidden可能是tuple了，要改
            # val = self.hidden[i][j-1][k].pop().move(dst)
            tmp = self.hidden[i][j-1][k].pop()
            if isinstance(tmp, tuple):
                val = []
                for v in tmp:
                    val.append(v.move(dst))
            else:
                val = tmp.move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                # XXX: hidden可能是tuple了，要改
                # x.val = x.val.move(self.act_home)
                if isinstance(x.val, tuple):
                    x.val = list(x.val)
                    for i in range(len(x.val)):
                        x.val[i] = x.val[i].move(self.act_home)
                    x.val = tuple(x.val)
                else:
                    x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()
