#!/usr/bin/env python3
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import math
import logging
import shutil
import subprocess
import gc
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def _image_files(folder: Path) -> List[Path]:
    return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

def prepare_concept(training_dir: Path, concept_name: str) -> Path:
    # Look for source folder
    concept_folder = training_dir / concept_name / "source"
    if not concept_folder.exists() or not concept_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {concept_folder}")
        
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = list(concept_folder.iterdir())
    
    # 1. Convert to .jpg and remove other formats
    for f in all_files:
        if f.is_file() and f.suffix.lower() in image_extensions and f.suffix.lower() != ".jpg":
            try:
                img = Image.open(f)
                rgb_img = img.convert('RGB')
                new_path = f.with_suffix('.jpg')
                rgb_img.save(new_path, quality=95)
                f.unlink()  # Delete original
                logger.info(f"Converted {f.name} to JPG")
            except Exception as e:
                logger.warning(f"Could not convert {f}: {e}")
                
    # 2. Rename sequentially and create text labels
    # Re-read files to only get the .jpg ones
    jpg_files = sorted([f for f in concept_folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
    
    for idx, f in enumerate(jpg_files, 1):
        new_name = f"{concept_name}.{idx:02d}.jpg"
        new_path = concept_folder / new_name
        if f != new_path:
            # Handle name collisions
            if new_path.exists():
                temp_path = concept_folder / f"{concept_name}_temp_{idx:02d}.jpg"
                f.rename(temp_path)
                f = temp_path
            f.rename(new_path)
            logger.debug(f"Renamed {f.name} to {new_name}")
        else:
            new_path = f
            
        # Create corresponding .txt file
        txt_path = new_path.with_suffix(".txt")
        txt_path.write_text(concept_name, encoding="utf-8")
        
    # Clean up old txt files that might not match the new sequence
    for f in concept_folder.iterdir():
        if f.is_file() and f.suffix.lower() == ".txt":
            if not (concept_folder / f.name.replace(".txt", ".jpg")).exists():
                f.unlink()

    logger.info(f"Prepared dataset for concept '{concept_name}' with {len(jpg_files)} images.")
    return concept_folder

def launch_training(args, dataset_path: Path, lora_output_path: Path, concept_name: str, final_loras_dir: Path):
    cmd = [
        "accelerate", "launch", 
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        os.path.abspath(__file__),
        "--internal_training",
        "--dataset_base_path", str(dataset_path),
        "--output_path", str(lora_output_path),
        "--model_path", args.model_path,
        "--concept", concept_name,
        "--num_epochs", str(args.num_epochs),
        "--train_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--resolution", str(args.resolution),
        "--num_workers", str(args.num_workers),
        "--lr_scheduler", str(args.lr_scheduler),
        "--resume_from_checkpoint", str(args.resume_from_checkpoint) if args.resume_from_checkpoint else "none"
    ]
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    logger.info("Starting training for '%s'", concept_name)
    logger.info("Training command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    # After successful training, copy the final safetensors
    final_safetensors = lora_output_path / "adapter_model.safetensors"
    if final_safetensors.exists():
        final_loras_dir.mkdir(parents=True, exist_ok=True)
        dest_path = final_loras_dir / f"{concept_name}.safetensors"
        shutil.copy(final_safetensors, dest_path)
        logger.info(f"Successfully copied final LoRA to {dest_path}")
    else:
        logger.error(f"Training completed but could not find final LoRA at {final_safetensors}")

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
    parser.add_argument("--training_dir", default="/root/.imgenie/train", help="Root folder with labeled image subfolders.")
    parser.add_argument("--output_dir", default="/root/.imgenie/loras", help="Output root for datasets and LoRA checkpoints.")
    parser.add_argument("--model_path", default="/root/.imgenie/models/TongyiMAI.ZImageTurbo", help="Path to the ZImageTurbo base model.")
    parser.add_argument("--concept", default=None, help="Optional single concept folder to train. If omitted, all concepts are trained.")
    parser.add_argument("--num_epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest")
    
    # Internal arguments for accelerate mode
    parser.add_argument("--internal_training", action="store_true", help="Internal flag used by accelerate launch")
    parser.add_argument("--dataset_base_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=4)
    
    # Advanced / Hyperparameters
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2, help="Lowered to 2 to save unified memory on APUs")
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM (disabled by default)")
    
    return parser.parse_args()


def save_lora_checkpoint(accelerator, transformer, optimizer, lr_scheduler, output_path):
    logger.info(f"Saving LoRA weights and state to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    
    # Export in Native Diffusers Format
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    
    state_dict = get_peft_model_state_dict(unwrapped_transformer)
    # Replace base_model.model with transformer so diffusers loads it natively
    native_state_dict = {k.replace("base_model.model.", "transformer."): v for k, v in state_dict.items()}
    
    save_file(native_state_dict, os.path.join(output_path, "adapter_model.safetensors"))
    
    # Save optimizer and scheduler manually to bypass massive accelerate.save_state memory overhead
    torch.save(optimizer.state_dict(), os.path.join(output_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(output_path, "scheduler.pt"))


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
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"],
    )
    transformer = get_peft_model(transformer, lora_config)
    
    # CRITICAL: PEFT initializes LoRA weights in float32. During forward pass, 
    # PEFT forces the massive activation tensors to upcast to float32 to match,
    # causing an immediate OOM. We must cast the LoRA weights to bfloat16!
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
            
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
            
            # Load LoRA weights manually
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sf_path = os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors")
            if os.path.exists(sf_path):
                sd = load_file(sf_path)
                # Map keys back to PEFT format
                peft_sd = {k.replace("transformer.", "base_model.model."): v for k, v in sd.items()}
                set_peft_model_state_dict(transformer, peft_sd)
                
            # Load Optimizer
            opt_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                
            # Load Scheduler
            sch_path = os.path.join(args.resume_from_checkpoint, "scheduler.pt")
            if os.path.exists(sch_path):
                lr_scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
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
                
        if (epoch + 1) % 20 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info(f"Generating validation samples and saving checkpoint at epoch {epoch + 1}...")
                
                ckpt_dir = os.path.join(args.output_path, "checkpoints", f"epoch_{epoch + 1:04d}")
                save_lora_checkpoint(accelerator, transformer, optimizer, lr_scheduler, ckpt_dir)
                saved_checkpoints.append(ckpt_dir)
                if len(saved_checkpoints) > 3:
                    oldest_ckpt = saved_checkpoints.pop(0)
                    logger.info(f"Removing oldest checkpoint: {oldest_ckpt}")
                    shutil.rmtree(oldest_ckpt, ignore_errors=True)
                
                transformer.eval()
                
                sample_dir = os.path.join(args.output_path, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                
                prompts = [
                    f"A professional photograph of {args.concept} in a studio setting.",
                ]
                
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                pipeline.transformer = unwrapped_transformer
                
                # Clear memory before heavy generation
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    for i, p in enumerate(prompts):
                        with torch.no_grad():
                            gen = torch.Generator(device="cuda").manual_seed(42)
                            image = pipeline(
                                prompt=p,
                                num_inference_steps=8,
                                height=args.resolution,
                                width=args.resolution,
                                guidance_scale=0.0,
                                generator=gen,
                                output_type="pil"
                            ).images[0]
                            image.save(os.path.join(sample_dir, f"epoch_{epoch + 1:04d}_sample_{i}.png"))
                except Exception as e:
                    logger.error(f"Failed to generate sample images: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                pipeline.transformer = transformer
                transformer.train()
                
                # Clear memory after generation
                gc.collect()
                torch.cuda.empty_cache()
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_checkpoint(accelerator, transformer, optimizer, lr_scheduler, args.output_path)

def main():
    args = parse_args()
    if args.resume_from_checkpoint == "none":
        args.resume_from_checkpoint = None

    # Setup file logging
    log_dir = Path(args.output_dir) if not args.internal_training else Path(args.output_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    if args.internal_training:
        run_training(args)
    else:
        training_dir = Path(args.training_dir)
        output_dir = Path(args.output_dir)
        final_loras_dir = output_dir / "characters"
        
        if not training_dir.exists():
            raise FileNotFoundError(f"Training data directory does not exist: {training_dir}")
            
        if args.concept:
            concept_folder = training_dir / args.concept
            if not concept_folder.exists() or not concept_folder.is_dir():
                raise FileNotFoundError(f"Concept folder not found: {concept_folder}")
            dataset_path = prepare_concept(training_dir, args.concept)
            lora_output_path = concept_folder / "training"
            launch_training(args, dataset_path, lora_output_path, args.concept, final_loras_dir)
        else:
            for concept_folder in sorted(training_dir.iterdir()):
                if concept_folder.is_dir():
                    try:
                        dataset_path = prepare_concept(training_dir, concept_folder.name)
                        lora_output_path = concept_folder / "training"
                        launch_training(args, dataset_path, lora_output_path, concept_folder.name, final_loras_dir)
                    except Exception as e:
                        logger.error(f"Failed to process {concept_folder.name}: {e}")

if __name__ == "__main__":
    main()
