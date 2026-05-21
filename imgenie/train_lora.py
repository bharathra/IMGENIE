#!/usr/bin/env python3
import os
import argparse
import math
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import ZImagePipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
# pyrefly: ignore [missing-import]
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def _image_files(folder: Path) -> List[Path]:
    return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

def prepare_concept(training_dir: Path, dataset_dir: Path, concept_folder: Path) -> Path:
    concept_name = concept_folder.name
    target_folder = dataset_dir / concept_name
    target_folder.mkdir(parents=True, exist_ok=True)

    images = _image_files(concept_folder)
    if not images:
        logger.warning("Skipping %s because it contains no supported image files.", concept_folder)
        return target_folder

    for image_path in images:
        dst_image = target_folder / image_path.name
        shutil.copy(image_path, dst_image)
        caption_path = target_folder / f"{image_path.stem}.txt"
        caption_path.write_text(concept_name, encoding="utf-8")
        logger.debug("Prepared %s and %s", dst_image, caption_path)

    logger.info("Prepared dataset for concept '%s' with %d images.", concept_name, len(images))
    return target_folder

def launch_training(args, dataset_path: Path, lora_output_path: Path, concept_name: str):
    cmd = [
        "accelerate", "launch", os.path.abspath(__file__),
        "--internal_training",
        "--dataset_base_path", str(dataset_path),
        "--output_path", str(lora_output_path),
        "--model_path", args.model_path,
        "--concept", concept_name,
        "--num_epochs", str(args.num_epochs),
        "--train_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--resume_from_checkpoint", str(args.resume_from_checkpoint) if args.resume_from_checkpoint else "none"
    ]
    logger.info("Starting training for '%s'", concept_name)
    logger.info("Training command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Finished training for '%s'. Output: %s", concept_name, lora_output_path)

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
    # User-facing wrapper arguments
    parser.add_argument("--training_dir", default="/root/.imgenie/training", help="Root folder with labeled image subfolders.")
    parser.add_argument("--output_dir", default="/root/.imgenie/loras", help="Output root for datasets and LoRA checkpoints.")
    parser.add_argument("--model_path", default="/root/.imgenie/models/TongyiMAI.ZImageTurbo", help="Path to the ZImageTurbo base model.")
    parser.add_argument("--concept", default=None, help="Optional single concept folder to train. If omitted, all concepts are trained.")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")
    
    # Internal arguments for accelerate mode
    parser.add_argument("--internal_training", action="store_true", help="Internal flag used by accelerate launch")
    parser.add_argument("--dataset_base_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=4)
    
    # Advanced / Hyperparameters
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    return parser.parse_args()


def save_lora_checkpoint(accelerator, transformer, output_path):
    logger.info(f"Saving LoRA weights and state to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_transformer.save_pretrained(
        save_directory=output_path,
        safe_serialization=True
    )
    accelerator.save_state(output_path)
    
    # Convert PEFT keys to standard Diffusers keys for compatibility
    from safetensors.torch import load_file, save_file
    sf_path = os.path.join(output_path, "adapter_model.safetensors")
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
    config_path = os.path.join(output_path, "adapter_config.json")
    if os.path.exists(config_path):
        os.remove(config_path)


def run_training(args):
    
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
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()
    
    dataset = ZImageDataset(args.dataset_base_path, args.concept, size=args.resolution)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_epochs * len(dataloader)
    )
    
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
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
    saved_checkpoints = []
    
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_dir = os.path.join(args.output_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
                    args.resume_from_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                else:
                    args.resume_from_checkpoint = None
            else:
                args.resume_from_checkpoint = None

        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            starting_epoch = int(os.path.basename(args.resume_from_checkpoint).split("_")[-1])
            global_step = starting_epoch * len(dataloader)
            
            # Repopulate saved_checkpoints list so rotation works
            checkpoint_dir = os.path.join(args.output_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
                checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
                saved_checkpoints = checkpoints
    
    logger.info(f"Starting training on {len(dataset)} images for {args.num_epochs} epochs. Starting from epoch {starting_epoch}...")
    for epoch in range(starting_epoch, args.num_epochs):
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
                lr_scheduler.step()
                optimizer.zero_grad()
                
            global_step += 1
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
                
        if (epoch + 1) % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info(f"Generating validation samples and saving checkpoint at epoch {epoch + 1}...")
                
                ckpt_dir = os.path.join(args.output_path, "checkpoints", f"epoch_{epoch + 1:04d}")
                save_lora_checkpoint(accelerator, transformer, ckpt_dir)
                saved_checkpoints.append(ckpt_dir)
                if len(saved_checkpoints) > 3:
                    oldest_ckpt = saved_checkpoints.pop(0)
                    logger.info(f"Removing oldest checkpoint: {oldest_ckpt}")
                    shutil.rmtree(oldest_ckpt, ignore_errors=True)
                
                transformer.eval()
                
                sample_dir = os.path.join(args.output_path, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                
                prompts = [
                    f"A photograph of {args.concept} in a studio",
                    f"A professional photograph of {args.concept} in a boat in in a beautiful lake flanked by small hills on either side with a setting sun in the background"
                ]
                
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                pipeline.transformer = unwrapped_transformer
                
                for i, p in enumerate(prompts):
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            image = pipeline(
                                prompt=p,
                                num_inference_steps=20,
                                guidance_scale=7.0,
                                output_type="pil"
                            ).images[0]
                            image.save(os.path.join(sample_dir, f"epoch_{epoch + 1:04d}_sample_{i}.png"))
                
                pipeline.transformer = transformer
                transformer.train()
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_checkpoint(accelerator, transformer, args.output_path)

def main():
    args = parse_args()
    if args.resume_from_checkpoint == "none":
        args.resume_from_checkpoint = None

    if args.internal_training:
        run_training(args)
    else:
        training_dir = Path(args.training_dir)
        output_dir = Path(args.output_dir)
        dataset_dir = output_dir / "datasets"
        
        if not training_dir.exists():
            raise FileNotFoundError(f"Training data directory does not exist: {training_dir}")
            
        if args.concept:
            concept_folder = training_dir / args.concept
            if not concept_folder.exists() or not concept_folder.is_dir():
                raise FileNotFoundError(f"Concept folder not found: {concept_folder}")
            dataset_path = prepare_concept(training_dir, dataset_dir, concept_folder)
            lora_output_path = output_dir / f"lora_{args.concept}"
            launch_training(args, dataset_path, lora_output_path, args.concept)
        else:
            for concept_folder in sorted(training_dir.iterdir()):
                if concept_folder.is_dir():
                    dataset_path = prepare_concept(training_dir, dataset_dir, concept_folder)
                    lora_output_path = output_dir / f"lora_{concept_folder.name}"
                    launch_training(args, dataset_path, lora_output_path, concept_folder.name)

if __name__ == "__main__":
    main()
