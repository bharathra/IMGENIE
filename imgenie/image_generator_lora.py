#!/usr/bin/env python3
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from diffusers import ZImagePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TXTxIMG:
    def __init__(self, output_dir: str = "/root/.imgenie/txt2img"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[ZImagePipeline] = None

    def load_model(self, model: str = "Tongyi-MAI/Z-Image-Turbo", lora_paths: Optional[list] = None) -> None:
        try:
            logger.info(f"Loading base model: {model}")
            self.pipeline = ZImagePipeline.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                local_files_only=True
            ).to("cuda:0")

            if lora_paths:
                self._load_loras(lora_paths)

            logger.info("Model and LoRAs loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def _load_loras(self, lora_paths: list):
        """Loads multiple LoRA adapters into the pipeline correctly from local files."""
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        for i, path_str in enumerate(lora_paths):
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

        # Set default weights
        weights = [1.0 / len(lora_paths)] * len(lora_paths)
        self.pipeline.set_adapters([f"adapter_{i}" for i in range(len(lora_paths))], adapter_weights=weights)

    def _load_loras_old(self, lora_paths: list):
        """Loads multiple LoRA adapters into the pipeline."""

        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        for i, path in enumerate(lora_paths):
            adapter_name = f"adapter_{i}"
            logger.info(f"Loading LoRA {i}: {path} as {adapter_name}")
            # Load the weights and assign a unique name
            self.pipeline.load_lora_weights(path, adapter_name=adapter_name)

        # Set default weights (0.5 each for 2 LoRAs to stay at 1.0 total)
        weights = [1.0 / len(lora_paths)] * len(lora_paths)
        self.pipeline.set_adapters([f"adapter_{i}" for i in range(len(lora_paths))], adapter_weights=weights)

    def generate(self,
                 prompt: str,
                 negative_prompt: str = "",
                 height: int = 720,
                 width: int = 720,
                 num_inference_steps: int = 9,
                 guidance_scale: float = 0,
                 lora_weights: Optional[list] = None) -> tuple[str, float]:

        # If specific weights are provided for this generation, apply them
        if lora_weights and self.pipeline:
            adapter_names = [f"adapter_{i}" for i in range(len(lora_weights))]
            self.pipeline.set_adapters(adapter_names, adapter_weights=lora_weights)

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

    def _get_timestamped_path(self, prompt: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).replace(" ", "_")
        return self.output_dir / f"{timestamp}_{safe_prompt}.png"


if __name__ == "__main__":
    server = TXTxIMG()

    # Example: List your LoRA .safetensors paths here
    my_loras = [
        # characters 0.7
        # "/root/.imgenie/loras/z_image_turbo_alex_d/z_image_turbo_alex_d.safetensors", # LoRA wt: 0.6; Trigger: Alexandra Daddario
        "/root/.imgenie/loras/z_image_turbo_tam_bhat/z_image_turbo_tam_bhat.safetensors", # LoRA wt: 0.6; Trigger: Tam_Bhat
        # poses 0.3
        # "/root/.imgenie/loras/downblouse/downblouse_pose.safetensors",
        # "/root/.imgenie/loras/turbopussy/TurboPussyZ_v1.safetensors",
        # "/root/.imgenie/loras/mystic/Mystic-XXX-ZIT-v3.safetensors",
    ]

    server.load_model(lora_paths=my_loras)

    while True:
        user_prompt = input("Prompt: ")
        if user_prompt.lower() == 'exit':
            break

        # Adjust weights here: e.g., 0.7 for character, 0.3 for pose
        path, t = server.generate(prompt=user_prompt, lora_weights=[0.6])
        print(f"Done in {t:.2f}ms: {path}")
