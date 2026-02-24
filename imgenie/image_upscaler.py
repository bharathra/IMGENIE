#!/usr/bin/env python3

import logging
from os import path
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "/root/.imgenie/models/stabilityai.stable-diffusion-x4-upscaler"

class ImageUpscalerSD:
    """Image upscaler using Stable Diffusion x4 Upscaler."""

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 input_dir: str = "/root/.imgenie/input",
                 output_dir: str = "/root/.imgenie/output"):
        
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.pipeline = None

    def load_model(self) -> bool:
        try:
            if self.pipeline is None:
                logger.info(f"Loading SD Upscaler model: {self.model_path}")
                self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                )
                # Ensure the model goes to the GPU (works for both CUDA and ROCm given previous settings)
                self.pipeline = self.pipeline.to("cuda")
                
                # Memory optimizations are CRITICAL for large inputs since SD upscaler scales resolution by 4x.
                # A 720x720 image becomes 2880x2880, which requires massive VRAM during the diffusion process.
                self.pipeline.enable_attention_slicing()
                
                # VAE tiling is required to decode massive images without running out of memory
                if hasattr(self.pipeline, 'enable_vae_tiling'):
                    self.pipeline.enable_vae_tiling()
                
                logger.info("SD Upscaler model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading SD Upscaler model: {e}")
            return False

    def unload_model(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SD Upscaler model unloaded and GPU cache cleared.")
        else:
            logger.warning("No SD Upscaler model to unload.")
        self.pipeline = None

    def upscale(self, 
                image: Image.Image, 
                prompt: str = "high resolution, high quality, highly detailed", 
                negative_prompt: str = "blurry, low quality, artifacts, noisy",
                num_inference_steps: int = 20,
                guidance_scale: float = 9.0,
                target_max_size: int = 1024) -> Image.Image:
        """
        Upscale the image using text guidance. Stable Diffusion x4 Upscaler inherently upscales by exactly 4x.
        Unlike GAN upscalers, it relies on a prompt to 'hallucinate' appropriate details.
        target_max_size caps the output resolution by resizing the input to 1/4th of the target before upscaling.
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        # The SD upscaler expects RGB images
        image = image.convert("RGB")
        logger.info(f"Starting SD upscaling. Original input size: {image.size}")
        
        # Because SD x4 strictly upscales by 4x, to get a max 1024x1024 output,
        # we must shrink the input to max 256x256 before feeding it to the pipeline.
        if target_max_size:
            target_input_size = target_max_size // 4
            # Keep aspect ratio
            width, height = image.size
            if max(width, height) > target_input_size:
                ratio = target_input_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                logger.info(f"Resizing input image down to {new_width}x{new_height} so output doesn't exceed {target_max_size}x{target_max_size}")
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        logger.info(f"Using prompt: '{prompt}'")

        with torch.no_grad():
            upscaled_image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        logger.info(f"Upscale complete. Final size: {upscaled_image.size}")
        return upscaled_image

    def upscale_from_path(self, 
                          image_path: str, 
                          prompt: str = "high resolution, high quality, highly detailed", 
                          negative_prompt: str = "blurry, low quality, artifacts, noisy") -> Optional[Image.Image]:
        try:
            img = Image.open(image_path).convert('RGB')
            return self.upscale(img, prompt=prompt, negative_prompt=negative_prompt)
        except Exception as e:
            logger.error(f"Error loading image for SD upscaling: {e}")
            raise e

    def save_image(self, image: Image.Image, prefix: str = "upscale_sd") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{timestamp}_{prefix}.png"
        image.save(output_path)
        logger.info(f"Saved SD upscaled image to {output_path}")
        return str(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Optionally allow passing a custom prompt as the second argument
        prompt = sys.argv[2] if len(sys.argv) > 2 else "high resolution, photorealistic, highly detailed"
        
        upscaler = ImageUpscalerSD()
        if upscaler.load_model():
            upscaled_img = upscaler.upscale_from_path(image_path, prompt=prompt)
            if upscaled_img:
                upscaler.save_image(upscaled_img)
            upscaler.unload_model()
    else:
        print("Usage: python3 image_upscaler_sd.py <path_to_image> [optional_prompt]")
