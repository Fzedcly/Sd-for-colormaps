# -*- coding: utf-8 -*-
"""
LoRA 推理脚本（适配 diffusers 0.35+ / torch 2.5+）
- 从 ./lora_colormap_output/pytorch_lora_weights.safetensors 加载 LoRA
- 生成示例色带图到 ./inference_outputs/
"""

import os, time, random
from typing import Optional
import torch
from diffusers import StableDiffusionPipeline

# ==== 基本配置（按需改） ====
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_DIR = "./lora_colormap_output"
LORA_FILE = "pytorch_lora_weights.safetensors"
OUT_DIR = "./inference_outputs"

PROMPT = "smooth gradient from deep blue to teal green for ocean current visualization, clean color map, high fidelity"
NEGATIVE_PROMPT = "text, watermark, logo, frame, noisy, artifacts, clutter"

NUM_STEPS = 30           # 采样步数
GUIDANCE = 7.5           # CFG scale
H = W = 512              # 输出分辨率（SD1.5 默认512）
SEED: Optional[int] = 12345  # 固定随机种子，None 则随机


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 选择 dtype：GPU 优先 float16；CPU 用 float32
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    # 加载基础模型
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    # 尝试加载 LoRA（优先用 pipeline API，失败回退到 unet）
    lora_loaded = False
    try:
        pipe.load_lora_weights(LORA_DIR, weight_name=LORA_FILE)
        lora_loaded = True
    except Exception:
        try:
            pipe.unet.load_attn_procs(LORA_DIR, weight_name=LORA_FILE)
            lora_loaded = True
        except Exception as e:
            raise RuntimeError(f"无法加载 LoRA 权重 {os.path.join(LORA_DIR, LORA_FILE)}: {e}")
    if not lora_loaded:
        raise RuntimeError("LoRA 未加载成功，请检查路径与文件名。")

    # 固定随机种子
    if SEED is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = SEED
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 生成
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=H,
        width=W,
        generator=generator,
    )
    image = result.images[0]

    # 保存
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(OUT_DIR, f"colormap_{ts}_seed{seed}.png")
    image.save(out_path)
    print(f"✅ 推理完成：{out_path}")
    print(f"Prompt: {PROMPT}")
    print(f"Seed:   {seed} | Steps: {NUM_STEPS} | CFG: {GUIDANCE} | Size: {W}x{H}")


if __name__ == "__main__":
    main()
