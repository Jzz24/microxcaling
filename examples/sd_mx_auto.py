import torch
from diffusers import StableDiffusionPipeline

from mx import finalize_mx_specs
from mx import mx_mapping

simulate_mx = True
torch.manual_seed(44)

if simulate_mx:
    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'int4',
        'a_elem_format': 'int4',
        'block_size': 64,
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

# 加载预训练的Stable Diffusion模型
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")

# 设置生成图像的提示
prompt = "a photo of an astronaut riding a horse on mars"

# 生成图像
image = pipe(prompt).images[0]

# 保存生成的图像
image.save("generated_image.png")