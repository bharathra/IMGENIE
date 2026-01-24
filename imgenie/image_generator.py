#!/usr/bin/env python3

from random import seed
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml as yf   # to avoid conflict with PyYAML

import torch
from diffusers import ZImagePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL = "/root/.cache/huggingface/hub/model--Tongyi-MAI--Z-Image-Turbo/"
DEFAULT_NEGATIVE_PROMPT = "deformed, distorted, disjointed, disfigured, blurry, vibrators, sex toys, fuzzy, morbid, mutilated, mutated anatomy, malformed anatomy, missing anatomy, fused anatomy, unnatural anatomy"


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
                local_files_only=True
            ).to("cuda:0")

            logger.info("Model and LoRAs loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def load_loras(self, loras: list, weights: Optional[list] = None) -> None:
        # Load LoRA adapters by their identifiers
        #
        """Loads multiple LoRA adapters into the pipeline correctly from local files."""
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        # check if the loras to load are different from the currently active ones
        if set(loras) == set(self._active_loras):
            logger.info("Requested LoRAs are already active. No changes made.")
            return

        # First, unload existing loras
        self.pipeline.unload_lora_weights()
        self._active_loras = []

        # load added loras
        for i, path_str in enumerate(loras):
            #
            path = Path(path_str)
            adapter_name = f"adapter_{i}"
            logger.info(f"Loading LoRA {i}: {path.name} as {adapter_name}")
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
        weights = [1.0 / len(loras)] * len(loras)
        self.pipeline.set_adapters([f"adapter_{i}" for i in range(len(loras))], adapter_weights=weights)

    def generate(self,
                 prompt: str,
                 negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 height: int = 720,
                 width: int = 720,
                 num_inference_steps: int = 10,
                 guidance_scale: float = 0,
                 seed: Optional[int] = None) -> tuple[str, float]:

        inference_start = time.time()
        try:

            if self.pipeline is None:
                raise ValueError("Model pipeline is not loaded.")

            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    # generator=torch.Generator("cuda").manual_seed(seed),
                    height=height,
                    width=width,
                )

            image = result.images[0]
            inference_time = (time.time() - inference_start) * 1000
            output_path = self._get_timestamped_path(prompt)
            image.save(output_path)
            return str(output_path), inference_time

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return str(e), 0.0

    def generate_from_config(self, config_path: str) -> tuple[str, float]:
        try:
            with open(config_path, 'r') as f:
                config = yf.safe_load(f)

            height = config.get('height', 720)
            width = config.get('width', 720)

            num_inference_steps = config.get('num_inference_steps', 10)
            guidance_scale = config.get('guidance_scale', 0)

            # seed = config.get('seed', None)
            # if seed is None:
            #     seed = int(time.time() * 1000) % 2**32
            #     logger.info(f"Using random seed: {seed}")

            prompt = config.get('prompt', '')

            negative_prompt = config.get('negative_prompt', '')
            if negative_prompt is None:
                negative_prompt = ''
            negative_prompt += DEFAULT_NEGATIVE_PROMPT
            # logger.info(f"Final negative prompt: {negative_prompt}")

            loras = config.get('loras', [])
            # logger.info(f"LoRAs: {loras}")
            weights = [weight for _, weight in loras] if loras else None
            # logger.info(f"LoRA weights: {weights}")
            lora_paths = [path for path, _ in loras] if loras else []
            if lora_paths:
                self.load_loras(lora_paths, weights)

            return self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale)

        except Exception as e:
            logger.error(f"Error in generate_from_config: {e}")
            return str(e), 0.0

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
