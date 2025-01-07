import torch

from mx import finalize_mx_specs
from mx import mx_mapping

from transformers import AutoTokenizer, LlamaForCausalLM

simulate_mx = True
model_path = '/mnt/llm/LLaMA3/Meta-Llama-3-8B/'

if simulate_mx:
    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'fp4',
        'a_elem_format': 'fp4',
        'block_size': 128,
        'bfloat': 16,
        'custom_cuda': True,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
    }
    mx_specs = finalize_mx_specs(mx_specs)

    # Auto-inject MX modules and functions
    # This will replace certain torch.nn.* and torch.nn.functional.*
    # modules/functions in the global namespace!
    mx_mapping.inject_pyt_ops(mx_specs)


# 加载预训练的LLaMA模型和分词器
model = LlamaForCausalLM.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入提示
prompt = "The capital of France is?"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

# 生成文本
generate_ids = model.generate(inputs.input_ids, max_length=256, do_sample=False)

# 解码生成的文本
generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(generated_text)