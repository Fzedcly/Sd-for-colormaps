# -*- coding: utf-8 -*-
"""
LoRA 训练脚本 (适配 diffusers 0.35+ / PEFT)，已修复:
- fp16 训练不稳定 -> 使用 AMP + bfloat16 (可回退) + 降 LR + 梯度裁剪，避免 NaN
- 新版 save_lora_weights 需显式传层 -> 直接导出 LoRA 参数为 safetensors
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig
from safetensors.torch import save_file

# ================= 配置 =================
MODEL_ID = "runwayml/stable-diffusion-v1-5"

DATA_DIR = "./datasets/colormaps"     # 按你的目录调整
OUT_DIR  = "./lora_colormap_output"

RES    = 512
BATCH  = 1
EPOCHS = 3
LR     = 5e-5            # 降一点，配合 AMP 更稳
RANK   = 4
ALPHA  = RANK
SCALE  = 0.18215         # SD1.x VAE latent 缩放

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 优先 bfloat16（Ada 支持），否则 fp16，CPU 用 fp32
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
elif DEVICE == "cuda":
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = torch.float32

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ================= 数据集 =================
class ColormapDS(Dataset):
    def __init__(self, root):
        self.items = []
        root = Path(root)
        self.tfm = T.Compose([
            T.Resize((RES, RES), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            img = d / "grad1.png"
            prf = d / "prompt.txt"
            if img.exists() and prf.exists():
                self.items.append((img, prf.read_text(encoding="utf-8").strip()))
        if not self.items:
            raise RuntimeError(f"没有找到数据：{root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, prompt = self.items[idx]
        img = Image.open(p).convert("RGB")
        return {"pixel": self.tfm(img), "prompt": prompt}


def extract_lora_state_dict(model: torch.nn.Module):
    """
    仅导出 LoRA 相关参数（名字里包含 'lora' 的权重），用于 safetensors 保存。
    适配 PEFT + diffusers 新版的 adapter 注入。
    """
    sd = model.state_dict()
    out = {k: v.detach().cpu() for k, v in sd.items() if "lora" in k.lower()}
    if not out:
        raise RuntimeError("未找到 LoRA 参数，确认 add_adapter 是否成功以及 target_modules 是否匹配。")
    return out


def main():
    ds = ColormapDS(DATA_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2 if DEVICE == "cuda" else 0,
        pin_memory=(DEVICE == "cuda"),
    )

    # ================= 管线 =================
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,      # 主模型权重保留 fp32，更稳
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)

    unet = pipe.unet
    vae  = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer

    # 冻结非 LoRA 权重
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 省显存
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # ================= 用 PEFT 给 U-Net 加 LoRA（新版推荐做法） =================
    unet_lora_cfg = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # SD1.x CrossAttention 名称
    )
    unet.add_adapter(unet_lora_cfg)

    # 只训练 LoRA 参数
    train_params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and AMP_DTYPE != torch.float32))

    num_train_steps = len(dl) * EPOCHS
    print(f"样本数={len(ds)}  总步数≈{num_train_steps}  设备={DEVICE}  amp_dtype={AMP_DTYPE}")

    # ================= 训练循环 =================
    for ep in range(EPOCHS):
        for step, batch in enumerate(dl, 1):
            imgs = batch["pixel"].to(DEVICE, dtype=torch.float32)  # 送入 VAE 前保持 fp32

            # 图像 -> VAE latent
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * SCALE
                latents = latents.to(DEVICE)

            # 文本 -> 编码（CLIP）
            tokens = tokenizer(
                list(batch["prompt"]),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                enc = text_encoder(tokens.input_ids.to(DEVICE))[0]  # last_hidden_state

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=DEVICE
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            opt.zero_grad(set_to_none=True)

            # AMP 前向，优先 bfloat16
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda"), dtype=AMP_DTYPE):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=enc).sample
                loss = F.mse_loss(model_pred.float(), noise.float())

            # AMP 反向 & 更新
            if DEVICE == "cuda":
                scaler.scale(loss).backward()
                # 梯度裁剪抑制爆炸
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()

            if step % 10 == 0:
                print(f"[ep {ep+1}/{EPOCHS}] step {step:04d}/{len(dl)}  loss={loss.item():.6f}")

            # 若仍出现 NaN，立刻中断并给出提示
            if torch.isnan(loss):
                raise RuntimeError(
                    "检测到 loss=NaN：请尝试将 RES 降到 384、LR 再降到 2e-5，或缩小 batch。"
                )

    # ================= 保存 LoRA (safetensors) =================
    os.makedirs(OUT_DIR, exist_ok=True)
    unet_lora = extract_lora_state_dict(unet)
    save_path = os.path.join(OUT_DIR, "pytorch_lora_weights.safetensors")
    save_file(unet_lora, save_path)
    print("🎉 训练完成，LoRA 已保存到:", save_path)


if __name__ == "__main__":
    main()
