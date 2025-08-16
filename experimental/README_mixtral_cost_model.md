# Mixtral Cost Model

这是为 Mixtral 8x7B MoE 模型设计的策略求解器，基于原有的 `cost_model.py` 改进而来，专门支持 Mixtral 模型的 Mixture of Experts (MoE) 架构。

## 功能特性

- **支持 Mixtral 8x7B MoE 架构**：考虑了专家网络的稀疏激活特性
- **内存优化策略**：智能分配权重、缓存和激活在 GPU、CPU 和 NVMe 存储之间
- **吞吐量优化**：寻找最优的批处理大小和内存分配策略
- **硬件约束感知**：根据实际硬件限制生成可行的执行策略

## 与原版 cost_model.py 的主要区别

### 1. 模型架构适配
- **权重计算**：考虑了 MoE 结构中的专家网络权重
  - 注意力权重：q_proj, k_proj, v_proj, o_proj
  - MoE 权重：gate + **所有专家网络** (在大批量场景下，所有专家都可能被激活)
  - 层归一化权重
  - **重要**：权重传输量按所有 8 个专家计算，而非仅激活的 2 个专家

### 2. 计算复杂度调整
- **预填充阶段**：注意力计算 + MoE 门控 + **所有专家**计算
- **生成阶段**：考虑大批量场景下所有专家都可能被不同 token 激活
- **重要**：计算量按所有专家计算，因为在大批量推理中每个专家都会被某些 token 使用

### 3. 内存使用模式
- **中间激活**：为**所有专家**预留中间激活内存空间
- **专家并行**：考虑所有专家同时计算时的内存峰值
- **KV Cache 优化**：正确计算 Grouped Query Attention (GQA) 的 KV Cache 大小
  - 使用 `num_key_value_heads * head_dim` (8 × 128 = 1024) 而非 `hidden_size` (4096)
  - KV Cache 大小为标准注意力的 1/4，显著节省内存

## 安装依赖

```bash
pip install pulp
```

## 使用方法

### 1. 寻找最优策略

```bash
python mixtral_cost_model.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                             --prompt-len 512 --gen-len 32 \
                             --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500
```

### 2. 评估指定策略

```bash
python mixtral_cost_model.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                             --prompt-len 512 --gen-len 32 \
                             --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500 \
                             --gpu-batch-size 16 --num-gpu-batches 2 \
                             --percent 20 80 0 100 0 100
```

### 3. 编程接口

```python
from experimental.mixtral_cost_model import get_mixtral_config, solve_lp, solve
from flexllmgen.utils import GB

# 获取模型配置
config = get_mixtral_config("mistralai/Mixtral-8x7B-Instruct-v0.1")

# 设置工作负载参数
config.s = 512  # 提示长度
config.n = 32   # 生成长度

# 设置硬件约束
config.gmem = 16 * GB    # GPU 内存
config.cmem = 200 * GB   # CPU 内存
config.nmem = 1500 * GB  # NVMe 存储

# 寻找最优策略
args = {
    "gbs": None, "num_gb": None, "percent": None,
    "wg": None, "wc": None, "cg": None, "cc": None, "hg": None, "hc": None,
    "compress_w": False
}

best_policy, max_throughput = solve(config, solve_lp, args)
print(f"最大吞吐量: {max_throughput:.2f} token/s")
print(f"最优策略: {best_policy}")
```

## 参数说明

### 命令行参数

- `--model`: 模型名称 (目前支持 `mistralai/Mixtral-8x7B-Instruct-v0.1`)
- `--prompt-len`: 提示序列长度
- `--gen-len`: 生成序列长度
- `--gpu-mem`: GPU 内存大小 (GB)
- `--cpu-mem`: CPU 内存大小 (GB)
- `--nvme-mem`: NVMe 存储大小 (GB)
- `--gpu-batch-size`: GPU 批处理大小 (可选)
- `--num-gpu-batches`: GPU 批次数量 (可选)
- `--percent`: 分配百分比 `[wg, wc, cg, cc, hg, hc]` (可选)
  - `wg, wc`: 权重在 GPU/CPU 上的百分比
  - `cg, cc`: 缓存在 GPU/CPU 上的百分比  
  - `hg, hc`: 激活在 GPU/CPU 上的百分比
- `--compress-w`: 启用权重压缩
- `--alpha-g, --alpha-c, --alpha-n`: 内存约束放松比例

### 配置参数

```python
@dataclasses.dataclass
class MixtralCostModelConfig:
    # 工作负载参数
    s: int = 512          # 序列长度
    n: int = 32           # 生成长度
    
    # 模型架构参数
    l: int = 32           # 隐藏层数
    h1: int = 4096        # 隐藏维度
    h2: int = 14336       # 中间层维度
    nh: int = 32          # 注意力头数
    nkv: int = 8          # KV 头数
    head_dim: int = 128   # 头维度
    
    # MoE 特定参数
    num_experts: int = 8           # 专家数量
    num_experts_per_tok: int = 2   # 每个 token 激活的专家数
    
    # 硬件约束
    gmem: int = 15 * GB    # GPU 内存
    cmem: int = 204 * GB   # CPU 内存
    nmem: int = 1500 * GB  # NVMe 存储
```

## 输出解释

### 策略输出示例

```
Policy(gpu_batch_size=64, num_gpu_batches=9, 
       w_gpu_percent=0.54, w_cpu_percent=0.0, 
       cache_gpu_percent=0.0, cache_cpu_percent=0.99, 
       act_gpu_percent=0.0, act_cpu_percent=1.0, ...)
```

- `gpu_batch_size`: GPU 上的批处理大小
- `num_gpu_batches`: GPU 批次数量
- `w_gpu_percent/w_cpu_percent`: 权重在 GPU/CPU 上的分配比例
- `cache_gpu_percent/cache_cpu_percent`: 缓存在 GPU/CPU 上的分配比例
- `act_gpu_percent/act_cpu_percent`: 激活在 GPU/CPU 上的分配比例

### 性能指标

```
throughput = 23.56 token/s
gpu peak mem (prefill): 12.800 GB / 16.000 GB
gpu peak mem (gen): 7.308 GB / 16.000 GB
cpu peak mem (prefill): 145.451 GB / 200.000 GB
cpu peak mem (gen): 151.890 GB / 200.000 GB
```

## 示例程序

运行 `example_mixtral_usage.py` 查看详细的使用示例：

```bash
python experimental/example_mixtral_usage.py
```

## 注意事项

1. **硬件常数校准**：需要根据具体硬件环境调整 `MixtralCostModelConfig` 中的硬件常数
2. **内存约束**：合理设置 `alpha_g`, `alpha_c`, `alpha_n` 参数来控制内存约束的严格程度
3. **模型支持**：目前仅支持 `mistralai/Mixtral-8x7B-Instruct-v0.1`，可以扩展支持其他 Mixtral 变体
4. **量化支持**：压缩功能尚不完整，建议在生产环境中谨慎使用
5. **模型大小**：修正后的模型权重大小约为 86.25 GB（包含所有专家，fp16），正确反映了大批量推理的实际需求
6. **KV Cache 优化**：使用 GQA 的正确计算方式，KV Cache 大小为标准注意力的 1/4

## 与 FlexLLMGen 集成

生成的策略可以直接用于 FlexLLMGen 的 Mixtral 模型执行：

```python
# 使用生成的策略运行 Mixtral 模型
from flexllmgen.flex_moe import run_flexllmgen

# 将策略转换为命令行参数
args.gpu_batch_size = best_policy.gpu_batch_size
args.num_gpu_batches = best_policy.num_gpu_batches
args.percent = [
    int(best_policy.w_gpu_percent * 100),
    int(best_policy.w_cpu_percent * 100),
    int(best_policy.cache_gpu_percent * 100),
    int(best_policy.cache_cpu_percent * 100),
    int(best_policy.act_gpu_percent * 100),
    int(best_policy.act_cpu_percent * 100)
]

run_flexllmgen(args)
```