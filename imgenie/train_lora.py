#!/usr/bin/env python3
import os
import argparse
import math
import logging
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import ZImagePipeline
from peft import LoraConfig, get_peft_model
# pyrefly: ignore [missing-import]
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZImageDataset(Dataset):
    def __init__(self, data_dir, concept, size=720):
        self.data_dir = Path(data_dir)
        self.concept = concept
        self.size = size
        
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_paths.extend(list(self.data_dir.glob(f'*{ext}')))
        self.image_paths.extend(list(self.data_dir.glob(f'*{ext.upper()}')))
            
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "prompt": self.concept
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    logger.info(f"Loading ZImagePipeline from {args.model_path}")
    pipeline = ZImagePipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    
    transformer = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()
    
    dataset = ZImageDataset(args.dataset_base_path, args.concept, size=720)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    vae.to(device, dtype=torch.bfloat16)
    text_encoder.to(device, dtype=torch.bfloat16)
    
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=args.concept, 
        device=device,
        do_classifier_free_guidance=False
    )
    
    global_step = 0
    num_train_timesteps = pipeline.scheduler.config.num_train_timesteps
    
    logger.info(f"Starting training on {len(dataset)} images for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            bsz = pixel_values.shape[0]
            
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device).long()
                
                t_01 = (timesteps / num_train_timesteps).to(latents.dtype).view(bsz, 1, 1, 1)
                noisy_latents = (1.0 - t_01) * latents + t_01 * noise
                batched_prompt_embeds = [prompt_embeds[0]] * bsz
                latent_model_input = noisy_latents.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))
                timestep_model_input = (num_train_timesteps - timesteps) / num_train_timesteps
                timestep_model_input = timestep_model_input.to(dtype=torch.bfloat16)

            with accelerator.accumulate(transformer):
                model_out_list = transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    batched_prompt_embeds
                )[0]
                
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred
                
                target = (noise - latents).float()
                loss = F.mse_loss(noise_pred.float(), target, reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Saving LoRA weights to {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(
            save_directory=args.output_path,
            safe_serialization=True
        )
        
        # Convert PEFT keys to standard Diffusers keys for compatibility
        from safetensors.torch import load_file, save_file
        sf_path = os.path.join(args.output_path, "adapter_model.safetensors")
        sd = load_file(sf_path)
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("base_model.model."):
                new_k = "transformer." + k[len("base_model.model."):]
                new_sd[new_k] = v
            else:
                new_sd[k] = v
        save_file(new_sd, sf_path)
        
        # Remove adapter_config.json so Diffusers uses the standard loader
        config_path = os.path.join(args.output_path, "adapter_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    main()
