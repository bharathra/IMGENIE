#!/usr/bin/env python3

import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

import torch
from diffusers import ZImagePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TXTxIMG:
    """Wrapper for image generation pipelines with benchmarking."""

    def __init__(self,  # model: str = "Tongyi-MAI/Z-Image-Turbo",
                 model: str|None = None,
                 output_dir: str = "/root/.imgenie/txt2img"):
        # Store model identifier
        self.model = model
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        # Load the pipeline
        try:
            load_start = time.time()
            logger.info(f"Loading model: {self.model}")

            # 1. Use bfloat16 to prevent black images (NaNs)
            # 2. Use variant="fp16" to keep the initial download/load small
            self.pipeline = ZImagePipeline.from_pretrained(self.model,
                                                           torch_dtype=torch.bfloat16,
                                                           use_safetensors=True,
                                                           local_files_only=True)
            self.pipeline = self.pipeline.to("cuda:0")
            # 3. Memory optimizations to prevent the "end of generation" slowdown
            # self.pipeline.enable_model_cpu_offload()  # Efficiently moves parts of model
            # self.pipeline.vae.enable_tiling()        # Saves VAE memory

            load_time = (time.time() - load_start) * 1000
            logger.info(f"Model loaded in {load_time:.2f}ms")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def generate(self,
                 prompt: str,
                 negative_prompt: str = "",
                 height: int = 720,
                 width: int = 720,
                 num_inference_steps: int = 9,
                 guidance_scale: float = 0) -> tuple[str, float]:

        inference_start = time.time()
        logger.info(f"Generating image...")

        try:
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                )

            # Extract image from pipeline result
            if hasattr(result, 'images'):
                image = result.images[0]  # type: ignore
            else:
                image = result[0]  # For some versions, result is a tuple

            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image)
            elif isinstance(image, torch.Tensor):
                image = PILImage.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

            inference_time = (time.time() - inference_start) * 1000

            # Save with timestamped filename
            output_path = self._get_timestamped_path(prompt)
            image.save(output_path)  # type: ignore
            logger.info(f"Image saved to: {output_path}")

            return str(output_path), inference_time

        except RuntimeError as e:
            # Handle AMD GPU kernel compilation errors
            if 'HIP' in str(e) or 'hipError' in str(e):
                logger.error(f"GPU kernel error: {e}")
            return "GPU kernel error during inference.", 0.0

    def _get_timestamped_path(self, prompt: str) -> Path:
        """Generate a timestamped filename based on the prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from the prompt (first 30 chars)
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_'))
        safe_prompt = safe_prompt.replace(" ", "_").strip("_")
        safe_prompt = safe_prompt[:30] if safe_prompt else "image"
        filename = f"{timestamp}_{safe_prompt}.png"
        return self.output_dir / filename


if __name__ == "__main__":
    # Load model and wait for user prompt. Generate image and save to disk. And wait for next prompt until user exits.    
    server = TXTxIMG(model="Tongyi-MAI/Z-Image-Turbo",  # /root/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/
                     output_dir="/root/.imgenie/txt2img")
    server.load_model()

    while True:
        user_prompt = input("Enter prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
        output_path, inference_time = server.generate(prompt=user_prompt)
        print(f"Image generated and saved to {output_path} in {inference_time:.2f}ms")
