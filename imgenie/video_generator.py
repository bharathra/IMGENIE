#!/usr/bin/env python3

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# CogVideoX-5B Image-to-Video is a very capable open-source I2V model
DEFAULT_MODEL_PATH = "THUDM/CogVideoX-5b-I2V"


class VideoGenerator:
    """Image-to-Video generator using reference image and text prompts."""

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 output_dir: str = "/root/.imgenie/output/video"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.pipeline: Optional[CogVideoXImageToVideoPipeline] = None

    def load_model(self) -> bool:
        try:
            logger.info(f"Loading base video model: {self.model_path}")
            # Loading in bfloat16 to optimize for 64GB unified memory on AMD APU
            self.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16
            ).to("cuda:0") # Assuming ROCm translates "cuda:0" to the iGPU
            
            # Since AMD has 64GB VRAM, we don't necessarily need memory saving optimizations 
            # like enable_model_cpu_offload(), but VAE slicing could help avoid spikes
            self.pipeline.vae.enable_slicing()
            self.pipeline.vae.enable_tiling()
            
            logger.info("Video Model loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def unload_model(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()
            logger.info("Video Model pipeline unloaded and GPU cache cleared.")
        else:
            logger.warning("No video model pipeline to unload.")
        self.pipeline = None

    def _load_reference_image(self, image_path: str) -> Image.Image:
        """Load and validate reference image."""
        try:
            img = Image.open(image_path).convert('RGB')
            # CogVideoX-5B-I2V performs best around 720x480 according to spec
            # We'll smoothly resize while ignoring aspect ratio here for the test, 
            # though you'd likely want to crop or pad in production.
            target_width, target_height = 720, 480
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            raise e

    def generate(self,
                 prompt: str,
                 ref_image_path: str,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 6.0,
                 seed: Optional[int] = None,
                 num_frames: int = 49,
                 fps: int = 8):
        """
        Generate video based on reference image and text prompt.
        """
        try:
            if self.pipeline is None:
                raise ValueError("Model pipeline is not loaded.")

            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            reference_img = self._load_reference_image(ref_image_path)

            logger.info(f"Starting video generation with prompt: '{prompt}'")
            with torch.no_grad():
                video_frames = self.pipeline(
                    prompt=prompt,
                    image=reference_img,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,                 # 49 frames is default for CogVideoX
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).frames[0]

            logger.info("Video diffusion complete. Exporting to MP4...")
            
            output_path = self._get_timestamped_path(prompt)
            export_to_video(video_frames, str(output_path), fps=fps)
            logger.info(f"Saved video to {output_path}")
            
            return str(output_path)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def _get_timestamped_path(self, prompt: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).replace(" ", "_")
        return self.output_dir / f"{timestamp}_{safe_prompt}.mp4"


if __name__ == "__main__":
    import sys
    
    # Simple CLI tool to test it out
    if len(sys.argv) < 3:
        print("Usage: python3 video_generator.py <path_to_image> \"<prompt>\"")
        sys.exit(1)
        
    image_path = sys.argv[1]
    prompt = sys.argv[2]
    
    generator = VideoGenerator()
    if generator.load_model():
        try:
            generator.generate(prompt=prompt, ref_image_path=image_path)
            generator.unload_model()
        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            generator.unload_model()
