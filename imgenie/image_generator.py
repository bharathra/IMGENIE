#!/usr/bin/env python3

import logging
from os import path
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml as yf   # to avoid conflict with PyYAML

import torch
from random import seed
from diffusers import ZImagePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL = "/root/.cache/huggingface/hub/model--Tongyi-MAI--Z-Image-Turbo/"


class TXTxIMG:

    _active_loras: list = []

    def __init__(self, output_dir: str = "/root/.imgenie/txt2img"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[ZImagePipeline] = None

        self.load_model(MODEL)

    def load_model(self, model: str) -> None:
        try:
            logger.info(f"Loading base model: {model}")
            self.pipeline = ZImagePipeline.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                local_files_only=True).to("cuda:0")
            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def load_loras(self, loras: list, weights: Optional[list] = None) -> None:
        # Load LoRA adapters by their identifiers
        #
        """Loads multiple LoRA adapters into the pipeline correctly from local files."""
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        # First, unload existing loras
        if self._active_loras:
            self.pipeline.unload_lora_weights()
            self._active_loras = []

        if not loras:
            # logger.info("No LoRAs to load. Exiting load_loras.")
            return

        # load added loras
        for i, path_str in enumerate(loras):
            path = Path(path_str)
            adapter_name = f"adapter_{i}"
            # logger.info(f"Loading LoRA {i}: {path.name} as {adapter_name}")
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

    def generate(self,
                 prompt: str,
                 negative_prompt: str = "",
                 height: int = 720,
                 width: int = 720,
                 number_of_images: int = 1,
                 num_inference_steps: int = 10,
                 guidance_scale: float = 0,
                 seed: Optional[int] = None):

        try:

            if self.pipeline is None:
                raise ValueError("Model pipeline is not loaded.")

            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=number_of_images,
                    # generator=torch.Generator("cuda").manual_seed(seed),
                    height=height,
                    width=width,
                )

            for i, img in enumerate(result.images):
                output_path = self._get_timestamped_path(f"{prompt}_{i}")
                img.save(output_path)
                logger.info(f"Saved image {i} to {output_path}")

        except Exception as e:
            logger.error(f"Inference error: {e}")

    def generate_from_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config = yf.safe_load(f)

            height = config.get('height', 720)
            width = config.get('width', 720)

            number_of_images = config.get('number_of_images', 1)
            guidance_scale = config.get('guidance_scale', 0)
            num_inference_steps = config.get('num_inference_steps', 9)

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

            self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                number_of_images=number_of_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale)

        except Exception as e:
            logger.error(f"Error in generate_from_config: {e}")

    def _get_timestamped_path(self, prompt: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).replace(" ", "_")
        return self.output_dir / f"{timestamp}_{safe_prompt}.png"


if __name__ == "__main__":
    generator = TXTxIMG()

    # while True:
    #     user_prompt = input("Prompt: ")
    #     if user_prompt.lower() == 'exit':
    #         break
    #     path, t = server.generate(prompt=user_prompt, lora_weights=[0.7])  # Adjust weights as needed
    #     print(f"Done in {t:.2f}ms: {path}")

    while True:
        input("Press >Enter to generate image from config file...")
        generator.generate_from_config("/root/.imgenie/prompts/prompt.yaml")
