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

    def load_model(self, model: str = "/root/.cache/huggingface/hub/model--Tongyi-MAI--Z-Image-Turbo/", lora_paths: Optional[list] = None) -> None:
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
                 negative_prompt: str = "ugly, deformed, distorted, disfigured, lowres, bad anatomy, blurry, vibrators, sex toys, fuzzy, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands and fingers, poorly drawn hands and fingers, poorly drawn face, mutation, deformed, blurry, dehydrated, bad proportions, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
                 height: int = 720,
                 width: int = 720,
                 num_inference_steps: int = 12,
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
        # "/root/.imgenie/loras/z_image_turbo_tam_bhat/z_image_turbo_tam_bhat.safetensors", # LoRA wt: 0.6; Trigger: Tam_Bhat
        # "/root/.imgenie/loras/z_image_turbo_deb_ryan2/z_image_turbo_deb_ryan2.safetensors", # LoRA wt: 0.6; Trigger: deb_ryan
        # "/root/.imgenie/loras/z_image_turbo_kajol/z_image_turbo_kajol.safetensors", # LoRA wt: 0.6; Trigger: Kajol
        # "/root/.imgenie/loras/z_image_madhuri/z_image_madhuri.safetensors", # LoRA wt: 0.6; Trigger: Madhuri Dixit
        # "/root/.imgenie/loras/z_image_rachel/z_image_rachel.safetensors",  # LoRA wt: 0.6; Trigger: Rachel Weisz
        # "/root/.imgenie/loras/z_image_mila/z_image_mila.safetensors", # LoRA wt: 0.6; Trigger: Mila Kunis
        # "/root/.imgenie/loras/z_image_melissa/z_image_melissa.safetensors",  # LoRA wt: 0.6; Trigger: Melissa Joan Hart
        # "/root/.imgenie/loras/z_image_anna/z_image_anna.safetensors",  # LoRA wt: 0.6; Trigger: Anna Torv
        "/root/.imgenie/loras/z_image_rani_m/z_image_rani_m.safetensors",  # LoRA wt: 0.6; Trigger: Rani Mukerji
        # "/root/.imgenie/loras/z_image_kat_den/z_image_kat_den.safetensors",  # LoRA wt: 0.6; Trigger: Kat Dennings
        # "/root/.imgenie/loras/z_image_alicia_silver/z_image_alicia_silver.safetensors",  # LoRA wt: 0.6; Trigger: Alicia Silverstone
        # "/root/.imgenie/loras/z_image_amy_adams/z_image_amy_adams.safetensors",  # LoRA wt: 0.6; Trigger: Amy Adams
        # "/root/.imgenie/loras/z_image_naomi_scott/z_image_naomi_scott.safetensors",  # LoRA wt: 0.6; Trigger: Naomi Scott
        # "/root/.imgenie/loras/z_image_lindsay_lohan/z_image_lindsay_lohan.safetensors",  # LoRA wt: 0.7; Trigger: Lindsay Lohan
        # "/root/.imgenie/loras/z_image_amber_heard/z_image_amber_heard.safetensors",  # LoRA wt: 0.7; Trigger: Amber Heard
        # muliple loras in one folder
        # "/root/.imgenie/loras/characters/ZIdT Alexis Bledel (vrtlAlexisBledel).safetensors",  # LoRA wt: 0.7; Trigger: Alexis Bledel
        # "/root/.imgenie/loras/characters/ZIdT Alice Eve (vrtlAliceEve).safetensors",  # LoRA wt: 0.7; Trigger: Alice Eve
        # "/root/.imgenie/loras/characters/ZIdT Bryce Dallas Howard (vrtlBryceDallasHoward).safetensors",  # LoRA wt: 0.7; Trigger: Bryce Dallas Howard
        # "/root/.imgenie/loras/characters/ZIdT Carla Gugino (vrtlCarlaGugino).safetensors", # LoRA wt: 0.7; Trigger: Carla Gugino
        # "/root/.imgenie/loras/characters/ZIdT Hayley Atwell (vrtlHayleyAtwell).safetensors",  # LoRA wt: 0.7; Trigger: Hayley Atwell
        # "/root/.imgenie/loras/characters/ZIdT Hilary Duff (vrtlHilaryDuff).safetensors",  # LoRA wt: 0.7; Trigger: Hilary Duff
        # "/root/.imgenie/loras/characters/ZIdT Jessica Biel (vrtljessicabiel).safetensors",  # LoRA wt: 0.7; Trigger: Jessica Biel
        # "/root/.imgenie/loras/characters/ZIdT Joanne Kelly (vrtlJoanneKelly).safetensors",  # LoRA wt: 0.7; Trigger: Joanne Kelly
        # "/root/.imgenie/loras/characters/ZIdT Kira Kosarin (vrtlKiraKosarin).safetensors",  # LoRA wt: 0.7; Trigger: Kira Kosarin
        # "/root/.imgenie/loras/characters/ZIdT Kristen Bell (vrtlKristenBell).safetensors",  # LoRA wt: 0.7; Trigger: Kristen Bell
        # "/root/.imgenie/loras/characters/ZIdT Lili Reinhart (vrtlLiliReinhart).safetensors",  # LoRA wt: 0.7; Trigger: Lili Reinhart
        # "/root/.imgenie/loras/characters/ZIdT Michelle Trachtenberg (vrtlMichelleTrachtenberg).safetensors",  # LoRA wt: 0.7; Trigger: Michelle Trachtenberg
        # "/root/.imgenie/loras/characters/ZIdT Rachel McAdams (vrtlRachelMcAdams).safetensors",  # LoRA wt: 0.7; Trigger: Rachel McAdams
        # "/root/.imgenie/loras/characters/ZIdT Sarah Lancaster (vrtlSarahLancaster).safetensors",  # LoRA wt: 0.7; Trigger: Sarah Lancaster
        # "/root/.imgenie/loras/characters/ZIdT Sarah Silverman (vrtlSarahSilverman).safetensors",  # LoRA wt: 0.7; Trigger: Sarah Silverman      

        # poses 0.3
        # "/root/.imgenie/loras/gpussy/gp-zimage_000008000.safetensors",
        # "/root/.imgenie/loras/z_image_turbo_vulva_spread/z_image_turbo_vulva_spread.safetensors", # LoRA wt: 0.3; Trigger: vulva spread
    ]

    server.load_model(lora_paths=my_loras)

    while True:
        user_prompt = input("Prompt: ")
        if user_prompt.lower() == 'exit':
            break

        # Adjust weights here: e.g., 0.7 for character, 0.3 for pose
        path, t = server.generate(prompt=user_prompt, 
                                  lora_weights=[0.7])  # Adjust weights as needed
        print(f"Done in {t:.2f}ms: {path}")
