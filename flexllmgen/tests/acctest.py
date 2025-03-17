import os
import sys
import torch
import json
import numpy as np
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding

# 设置环境变量
os.environ["HF_HOME"] = "/data1/zzh/huggingface"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# 
import transformers.models.mixtral.modeling_mixtral
import test_mixtral_model
transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer = test_mixtral_model.MixtralDecoderLayer
transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = test_mixtral_model.MixtralSparseMoeBlock
# transformers.MixtralForCausalLM = test_mixtral_model.MixtralForCausalLM

# 加载模型
config = MixtralConfig.from_pretrained(MODEL_NAME)
model = test_mixtral_model.MixtralForCausalLM.from_pretrained(MODEL_NAME, config=config, torch_dtype=torch.float16, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 生成prompt
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
    input = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len, truncation=True, return_tensors="pt")
    return input["input_ids"], input["attention_mask"]

# 将prompt送入到模型中进行推理
inputs, attention_mask = get_test_inputs(2, 1, tokenizer)
print("warmup_inputs:", inputs)
print("attention_mask:", attention_mask)
# with open('output.txt', 'w') as f:
#     f.write(str(inputs))


class layeroutputhook: # Record the inputs and outputs of each operator in layer 1
    def __init__(self, model, open_hook=False):
        self.open_hook = open_hook
        if not open_hook: return
        self.embed_handle = []  
        self.pe_handle = []  
        self.attn_handle = [] 
        self.gate_handle = [] 
        self.moe_handle = []  
        # with open('output.txt', 'w') as f:
        #     for name, module in model.named_modules():
        #         f.write(name + '\n')
        for name, module in model.named_modules():
            if name == "model.embed_tokens":
                self.embed_handle.append(module.register_forward_hook(self.hook(name)))
                print(name)
            else:
                if len(name.split('.')) <= 2: continue
                layer = int(name.split('.')[2])
                if layer > 0: break
                # if name == "model.layers.0.self_attn":
                #     self.attn_handle.append(module.register_forward_hook(self.hook(name)))
                #     print(name)
                if name == "model.layers.0.block_sparse_moe.gate":
                    self.gate_handle.append(module.register_forward_hook(self.hook(name)))
                    print(name)
                if name == "model.layers.0.block_sparse_moe.experts":
                    self.moe_handle.append(module.register_forward_hook(self.hook(name)))
                    print(name)

    def hook(self, name):
        def hook_fn(module, input, output):
            torch.set_printoptions(profile="full")
            with open('output_' + name.split('.')[-1] + '.txt', 'w') as f:
                f.write(str(output))
        return hook_fn
    
    def close(self):
        self.log()
        if not self.open_hook:
            return 
        self.embed_handle.remove()
        self.attn_handle.remove()
        self.gate_handle.remove()
        self.moe_handle.remove()
layeroutputhook = layeroutputhook(model, open_hook=True)

# inputs = torch.tensor(inputs)

model.attention_mask = attention_mask
outputs = model.generate(inputs, max_new_tokens=1, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)

# 打印生成的结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
