# -*- coding: utf-8 -*-
"""
生成左→右“黄(#FFFF00) 到 红(#FF0000)”的鲜艳、清晰颜色表（色带）
- 强制 Img2Img + 灰度 ramp（锁定为 1D 渐变）
- 采样器：Euler Ancestral（对纯色带更稳）
- 关闭 Diffusers 隐形水印（避免蓝点）
- LoRA 加载带强校验（常见权重名 + fallback）
- 导出三份：原尺寸(256x16)、标准(256x16)、超细(256x1)

在 PyCharm：
- Script path：本文件
- Working directory：项目根或本文件所在目录
"""

import os
import numpy as np
from PIL import Image
import torch

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)

print("[RUNNING FILE]", __file__)  # 确认跑的是这份文件

# ========= 路径与尺寸 =========
MODEL_ID   = "runwayml/stable-diffusion-v1-5"

# ★★ 改成你的 LoRA 绝对路径（建议目录里包含 pytorch_lora_weights.safetensors）
LORA_DIR   = r"D:\Studying\Ms.zeng\SD\pythonProject\lora_colormap_output"

# ★★ 改成你的输出目录绝对路径
OUTPUT_DIR = r"D:\Studying\Ms.zeng\SD\pythonProject\inference_outputs"

# 目标横条尺寸（与需求一致）
WIDTH, HEIGHT = 256, 16
VERTICAL = False  # 竖条请对调宽高并设为 True

# ========= 采样与控制（更稳） =========
STEPS    = 30
GUIDANCE = 4.2          # 低 CFG，减弱语义幻觉
STRENGTH = 0.18         # 低去噪强度，强保留 ramp 形状；若上色弱可临时升至 0.35~0.50
SEED     = 123

# LoRA 影响力（颜色更“艳”）
LORA_WEIGHT = 1.6

DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 提示词（黄→红、鲜艳、清晰） =========
PROMPT = (
    "scientific colorbar, pure 1D horizontal color gradient, "
    "left to right: yellow (#FFFF00) to red (#FF0000), "
    "smooth, banding-free, uniform, high saturation, vivid, "
    "no texture, no pattern, no noise, no frame, no ticks"
)

NEGATIVE = (
    "photo, photographic, people, object, scene, background, "
    "world map, earth, continent, ocean, terrain, relief, topography, satellite, "
    "texture, pattern, grain, noise, artifacts, vertical stripes, streaks, bars, columns, "
    "grid, border, frame, labels, numbers, text, watermark, banding, posterization, blur"
)

# ========= 工具函数 =========
def make_ramp(w: int, h: int, vertical: bool = False):
    """生成灰度 ramp（L），复制到 RGB 通道作为 Img2Img 条件图；同时返回 L 便于自检保存。"""
    if vertical:
        arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    else:
        arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8)[None, :], (h, 1))
    imgL = Image.fromarray(arr).convert("L")               # 避免 Pillow 关于 mode 的弃用警告
    imgRGB = Image.merge("RGB", (imgL, imgL, imgL))
    return imgRGB, imgL

def seed_everything(seed: int):
    if seed is None or seed < 0:
        seed = torch.randint(0, 2**31-1, (1,)).item()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32 - 1))
    return seed

def try_load_lora(pipe, lora_dir: str, lora_scale: float = 1.2) -> bool:
    """常见权重名尝试 + 目录式 + fallback attn_procs；返回是否加载成功。"""
    loaded = False
    names = [
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "diffusers_lora.safetensors",
        "diffusers_lora.bin",
    ]
    for n in names:
        p = os.path.join(lora_dir, n)
        if os.path.exists(p):
            try:
                pipe.load_lora_weights(lora_dir, weight_name=n)
                pipe.fuse_lora(lora_scale=lora_scale)
                print(f"[INFO] LoRA loaded weight={n}, scale={lora_scale}")
                loaded = True
                break
            except Exception as e:
                print(f"[WARN] load {n} failed:", e)

    if not loaded:
        try:
            pipe.load_lora_weights(lora_dir, weight_name=None)
            pipe.fuse_lora(lora_scale=lora_scale)
            print(f"[INFO] LoRA loaded from dir, scale={lora_scale}")
            loaded = True
        except Exception as e:
            print("[WARN] load_lora_weights(dir) failed:", e)

    if not loaded:
        try:
            pipe.unet.load_attn_procs(lora_dir)
            if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "load_attn_procs"):
                try:
                    pipe.text_encoder.load_attn_procs(lora_dir)
                except Exception:
                    pass
            print(f"[INFO] Fallback attn_procs loaded, scale={lora_scale}")
            loaded = True
        except Exception as e:
            print("[ERR] fallback attn_procs failed:", e)

    print("[INFO] LoRA loaded flag ->", loaded)
    return loaded

def save_resized_variants(img: Image.Image, out_dir: str):
    """导出标准(256x16, LANCZOS) 与超细(256x1, BOX) 两份成品。"""
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "colormap_256x16.png")
    p2 = os.path.join(out_dir, "colormap_256x1.png")
    img.resize((256, 16), Image.LANCZOS).save(p1, compress_level=0)
    img.resize((256, 1),  Image.BOX).save(p2, compress_level=0)
    print("Saved:", p1)
    print("Saved:", p2)

# ========= 主流程 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, safety_checker=None
    )
    # 采样器：Euler Ancestral（更服从“纯渐变”的约束）
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    # 关闭隐形水印（防蓝点）
    if hasattr(pipe, "watermark"):
        pipe.watermark = None
    if hasattr(pipe, "set_watermark"):
        try:
            pipe.set_watermark(None)
        except Exception:
            pass

    ok = try_load_lora(pipe, LORA_DIR, lora_scale=LORA_WEIGHT)
    assert ok, "❌ LoRA 未成功加载：请检查 LORA_DIR 路径/文件名（建议使用绝对路径与 pytorch_lora_weights.safetensors）"

    # 生成 ramp 并保存调试图
    init_rgb, init_L = make_ramp(WIDTH, HEIGHT, vertical=VERTICAL)
    init_L_path = os.path.join(OUTPUT_DIR, "_debug_ramp.png")
    init_L.save(init_L_path, compress_level=0)
    print("[INFO] init ramp size =", init_rgb.size, "| saved:", init_L_path)

    seed = seed_everything(SEED)
    print("[INFO] Seed =", seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=init_rgb,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=gen,
    ).images[0]

    print("[INFO] output size =", out.size)
    assert out.size == init_rgb.size, "❌ 输出尺寸 != ramp 尺寸（Run 配置或流程有误）"

    base = f"colormap_seed{seed}_{WIDTH}x{HEIGHT}.png"
    p0 = os.path.join(OUTPUT_DIR, base)
    out.save(p0, compress_level=0)
    print("Saved:", p0)

    # 导出标准与超细两份
    save_resized_variants(out, OUTPUT_DIR)
    print("🎉 完成！")

if __name__ == "__main__":
    main()
