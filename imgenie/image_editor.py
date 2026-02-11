#!/usr/bin/env python3

import logging
from os import path
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml as yf   # to avoid conflict with PyYAML

import torch
from PIL import Image
from diffusers import ZImageImg2ImgPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL = "/root/.cache/huggingface/hub/model--Tongyi-MAI--Z-Image-Turbo/"


class IMGxIMG:
    """Image-to-Image editor using reference image and text prompts."""

    _active_loras: list = []

    def __init__(self, output_dir: str = "/root/.imgenie/img2img"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[ZImageImg2ImgPipeline] = None

        self.load_model(MODEL)

    def load_model(self, model: str) -> None:
        try:
            logger.info(f"Loading base model: {model}")
            self.pipeline = ZImageImg2ImgPipeline.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                local_files_only=True).to("cuda:0")
            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def load_loras(self, loras: list, weights: Optional[list] = None) -> None:
        """Loads multiple LoRA adapters into the pipeline correctly from local files."""
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        # First, unload existing loras
        if self._active_loras:
            self.pipeline.unload_lora_weights()
            self._active_loras = []

        if not loras:
            return

        # load added loras
        for i, path_str in enumerate(loras):
            path = Path(path_str)
            adapter_name = f"adapter_{i}"
            # We pass the PARENT directory as the first argument,
            # and the specific filename as weight_name.
            self.pipeline.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=adapter_name
            )

        self._active_loras = loras.copy()
        logger.info(f"Active LoRAs after update: {self._active_loras}")

        # Set default weights
        self.pipeline.set_adapters([f"adapter_{i}" for i in range(len(loras))], adapter_weights=weights)

    def _load_reference_image(self, image_path: str) -> Image.Image:
        """Load and validate reference image."""
        try:
            img = Image.open(image_path).convert('RGB')
            logger.info(f"Loaded reference image: {image_path}")

            # resize to 720p without changing aspect ratio
            width, height = img.size
            img = img.resize((720, int(720 * height / width)))
            # resize to multiple of 16
            width, height = img.size    
            new_width = width - (width % 16)
            new_height = height - (height % 16)
            if new_width != width or new_height != height:
                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height))            

            return img
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            raise e

    def edit(self,
             reference_image_path: str,
             prompt: str,
             negative_prompt: str = "",
             number_of_images: int = 1,
             num_inference_steps: int = 10,
             guidance_scale: float = 0,
             strength: float = 0.8,
             seed: Optional[int] = None):
        """
        Generate edited images based on reference image and text prompt.
        
        Args:
            reference_image_path: Path to the reference/input image
            prompt: Text prompt for editing guidance
            negative_prompt: Text prompt for what to avoid
            number_of_images: Number of variations to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: How much to follow the prompt
            strength: How much to modify the original image (0.0-1.0)
                     0 = no change, 1 = complete regeneration
            seed: Random seed for reproducibility
        """
        try:
            if self.pipeline is None:
                raise ValueError("Model pipeline is not loaded.")

            # Load reference image
            reference_img = self._load_reference_image(reference_image_path)
            
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    image=reference_img,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=number_of_images,
                    strength=strength,
                    height=reference_img.height,
                    width=reference_img.width,
                )

            for i, img in enumerate(result.images):
                output_path = self._get_timestamped_path(f"{prompt}_{i}")
                img.save(output_path)
                logger.info(f"Saved edited image {i} to {output_path}")

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def edit_from_config(self, config_path: str):
        """Edit images from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yf.safe_load(f)

            reference_image_path = config.get('reference_image_path', '')
            if not reference_image_path:
                raise ValueError("reference_image_path is required in config")

            number_of_images = config.get('number_of_images', 1)
            guidance_scale = config.get('guidance_scale', 0)
            num_inference_steps = config.get('num_inference_steps', 9)
            strength = config.get('strength', 0.8)

            prompt = config.get('prompt', '')
            negative_prompt = config.get('negative_prompt', '')
            lora_root_path = config.get('lora_root_path', '')

            lora_paths = []
            weights = []

            character = config.get('character', None)
            if character is not None:
                if '__CHARACTER__' in prompt:
                    prompt = prompt.replace('__CHARACTER__', character)
                #
                character_lora_file = character.replace(" ", "")
                char_lora_path = path.join(lora_root_path, 'characters',
                                           f"{character_lora_file}.safetensors")
                if path.exists(char_lora_path):
                    lora_paths.append(char_lora_path)
                    character_strength = config.get('character_strength', 0.66)
                    weights.append(character_strength)

            concept = config.get('concept', None)
            if concept is not None:
                if '__CONCEPT__' in prompt:
                    prompt = prompt.replace('__CONCEPT__', concept)
                #
                concept_lora_file = concept.replace(" ", "")
                concept_lora_path = path.join(lora_root_path, 'concepts',
                                              f"{concept_lora_file}.safetensors")
                if path.exists(concept_lora_path):
                    lora_paths.append(concept_lora_path)
                    concept_strength = config.get('concept_strength', 0.33)
                    weights.append(concept_strength)

            self.load_loras(lora_paths, weights)

            self.edit(
                reference_image_path=reference_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                number_of_images=number_of_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength)

        except Exception as e:
            logger.error(f"Error in edit_from_config: {e}")
            raise e

    def _get_timestamped_path(self, prompt: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).replace(" ", "_")
        return self.output_dir / f"{timestamp}_{safe_prompt}.png"


if __name__ == "__main__":
    editor = IMGxIMG()

    while True:
        input("Press Enter to edit image from config file...")
        editor.edit_from_config("/root/.imgenie/prompts/edit_prompt.yaml")
