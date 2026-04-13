#!/usr/bin/env python3

import logging
from os import path
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import yaml as yf   # to avoid conflict with PyYAML

import torch
from PIL import Image
from diffusers import (
    ZImageImg2ImgPipeline,
    ZImagePipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel
)
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "/root/.imgenie/models/TongyiMAI.ZImageTurbo/"


class ImageGenerator:
    """Image-to-Image editor using reference image and text prompts."""

    _active_loras: list = []

    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 input_dir: str = "/root/.imgenie/input",
                 output_dir: str = "/root/.imgenie/output", 
                 lora_path: str = "/root/.imgenie/loras",
                 base_model_path: Optional[str] = None):

        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_path = Path(lora_path)
        self.model_path = model_path
        self.base_model_path = base_model_path  # Base model for UNet-only checkpoints
        self.is_single_file_model = False  # Track if model is loaded from .safetensors
        self.is_unet_only = False  # Track if .safetensors contains only UNet weights

        self.pipeline: Optional[Union[ZImageImg2ImgPipeline, ZImagePipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]] = None

    def load_model(self) -> bool:
        try:
            logger.info(f"Loading base model: {self.model_path}")
            
            # Check if model is a single .safetensors file
            if self.model_path.endswith('.safetensors'):
                logger.info("Detected single .safetensors file. Analyzing structure...")
                
                try:
                    state_dict = load_file(self.model_path)
                    keys_list = list(state_dict.keys())
                    
                    # Detect model type based on key structure
                    has_context_refiner = any('context_refiner' in k for k in keys_list)
                    has_noise_refiner = any('noise_refiner' in k for k in keys_list)
                    has_model_diffusion = any('model.diffusion_model' in k for k in keys_list)
                    has_std_unet = any(k.startswith(('unet.', 'down_blocks.', 'up_blocks.', 'conv_in.')) for k in keys_list)
                    
                    if (has_context_refiner or has_noise_refiner or has_model_diffusion) and not has_std_unet:
                        # This is a ZImage model checkpoint - need to load it with a base ZImage model
                        logger.info("Detected ZImage model checkpoint. Loading with base ZImage model...")
                        
                        if not self.base_model_path:
                            logger.error(
                                f"❌ ZImage checkpoint detected but no base_model_path specified. "
                                f"For ZImage models (.safetensors with custom transformer), you must provide a base ZImage model path. "
                                f"\n\nExample configuration:\n"
                                f"  model_path: '/path/to/custom_model.safetensors'\n"
                                f"  base_model_path: '/root/.imgenie/models/TongyiMAI.ZImageTurbo/'\n"
                                f"\nModel path: {self.model_path}"
                            )
                            return False
                        
                        logger.info(f"Loading base ZImage model from: {self.base_model_path}")
                        # Load the base ZImage model
                        self.pipeline = ZImageImg2ImgPipeline.from_pretrained(
                            self.base_model_path,
                            torch_dtype=torch.bfloat16,
                            use_safetensors=True,
                            local_files_only=True
                        ).to("cuda:0")
                        
                        # Replace the transformer (diffusion model) with custom checkpoint
                        logger.info("Loading custom transformer model from safetensors...")
                        # For ZImage models, the state dict keys are prefixed with 'model.diffusion_model.'
                        # We need to strip this prefix to load into the pipeline's transformer
                        mapped_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('model.diffusion_model.'):
                                new_key = key.replace('model.diffusion_model.', '')
                                mapped_state_dict[new_key] = value
                            else:
                                # Keep keys as-is if they don't have the prefix
                                mapped_state_dict[key] = value
                        
                        self.pipeline.transformer.load_state_dict(mapped_state_dict, strict=False)
                        self.is_single_file_model = True
                        self.is_unet_only = True
                        logger.info("Model loaded successfully with custom transformer weights.")
                        return True
                    
                    elif has_std_unet:
                        # This is a Stable Diffusion UNet-only checkpoint
                        logger.info("Detected Stable Diffusion UNet-only checkpoint.")
                        
                        if not self.base_model_path:
                            logger.warning("No base_model_path specified for UNet-only checkpoint. Using default Stable Diffusion v1.5")
                            self.base_model_path = "runwayml/stable-diffusion-v1-5"
                        
                        logger.info(f"Loading base model: {self.base_model_path}")
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            self.base_model_path,
                            torch_dtype=torch.bfloat16,
                            local_files_only=False
                        ).to("cuda:0")
                        
                        # Load and replace the UNet
                        logger.info("Loading custom UNet from safetensors...")
                        unet = UNet2DConditionModel.from_config(self.pipeline.unet.config)
                        unet.load_state_dict(state_dict)
                        self.pipeline.unet = unet
                        
                        self.is_single_file_model = True
                        self.is_unet_only = True
                        logger.info("Model loaded successfully with custom UNet.")
                        return True
                    else:
                        # Try to load as full pipeline
                        logger.info("Attempting to load as full StableDiffusionPipeline...")
                        self.pipeline = StableDiffusionPipeline.from_single_file(
                            self.model_path,
                            torch_dtype=torch.bfloat16,
                            use_safetensors=True
                        ).to("cuda:0")
                        self.is_single_file_model = True
                        self.is_unet_only = False
                        logger.info("Model loaded successfully.")
                        return True
                
                except Exception as e:
                    logger.error(f"Error loading .safetensors file: {e}")
                    return False
            
            else:
                # Load from directory (ZImage model structure)
                logger.info("Loading model from directory structure")
                self.pipeline = ZImageImg2ImgPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True).to("cuda:0")
                self.is_single_file_model = False
            
            logger.info("Model loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def unload_model(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()
            logger.info("Model pipeline unloaded and GPU cache cleared.")
        else:
            logger.warning("No model pipeline to unload.")
        self.pipeline = None

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
            # logger.info(f"Loaded reference image: {image_path}")

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

    def generate(self,
                 prompt: str,
                 ref_image_path: Optional[str] = None,
                 negative_prompt: str = "",
                 num_inference_steps: int = 10,
                 guidance_scale: float = 0,
                 strength: float = 0.8,
                 seed: Optional[int] = None,
                 height: int = 720,
                 width: int = 720,
                 callback = None):
        """
        Generate edited images based on reference image and text prompt.

        Args:
            ref_image_path: Path to the reference/input image
            prompt: Text prompt for editing guidance
            negative_prompt: Text prompt for what to avoid
            number_of_images: Number of variations to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: How much to follow the prompt
            strength: How much to modify the original image (0.0-1.0)
                     0 = no change, 1 = complete regeneration
            seed: Random seed for reproducibility
            callback: Optional callback function for progress tracking
        """

        try:
            if self.pipeline is None:
                raise ValueError("Model pipeline is not loaded.")

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)

            # Load reference image if provided
            if ref_image_path and path.exists(ref_image_path):
                reference_img = self._load_reference_image(ref_image_path)
                height = reference_img.height
                width = reference_img.width
                
                with torch.no_grad():
                    # Check the actual pipeline type to determine which inference path to use
                    if isinstance(self.pipeline, (ZImageImg2ImgPipeline, ZImagePipeline)):
                        # ZImage model (directory-based or checkpoint-based)
                        result = self.pipeline(
                            prompt=prompt,
                            image=reference_img,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=1,
                            strength=strength,
                            height=height,
                            width=width,
                            generator=generator,
                            callback_on_step_end=callback
                        )
                    else:
                        # StableDiffusion model (img2img)
                        img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipeline.components)
                        result = img2img_pipe(
                            prompt=prompt,
                            image=reference_img,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=1,
                            strength=strength,
                            generator=generator,
                            callback_on_step_end=callback
                        )
            else:
                # Text-to-Image (no reference image provided)
                # logger.info("No reference image provided. Switching to Text-to-Image mode.")
                
                with torch.no_grad():
                    # Check the actual pipeline type to determine which inference path to use
                    if isinstance(self.pipeline, (ZImageImg2ImgPipeline, ZImagePipeline)):
                        # ZImage model (directory-based or checkpoint-based)
                        txt2img_pipe = ZImagePipeline(**self.pipeline.components)
                        result = txt2img_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=1,
                            height=height,
                            width=width,
                            generator=generator,
                            callback_on_step_end=callback
                        )
                    else:
                        # StableDiffusion model (txt2img)
                        result = self.pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=1,
                            height=height,
                            width=width,
                            generator=generator,
                            callback_on_step_end=callback
                        )

            return result.images[0]

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def generate_from_yaml(self, yaml_path: str) -> bool:
        """Edit images from configuration file."""
        try:
            with open(yaml_path, 'r') as f:
                config = yf.safe_load(f)

            seed = config.get('seed', None)
            height = config.get('height', 720)
            width = config.get('width', 720)

            number_of_images = config.get('number_of_images', 1)
            guidance_scale = config.get('guidance_scale', 0)
            num_inference_steps = config.get('num_inference_steps', 9)

            prompt = config.get('prompt', '')
            negative_prompt = config.get('negative_prompt', '')

            ref_image_path = config.get('ref_image_path', None)
            ref_image_strength = config.get('ref_image_strength', 0.8)

            lora_paths = []
            weights = []

            character = config.get('character', None)
            if character is not None:
                if '__CHARACTER__' in prompt:
                    prompt = prompt.replace('__CHARACTER__', character)
                #
                character_lora_file = character.replace(" ", "")
                char_lora_path = path.join(self.lora_path, 'characters',
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
                concept_lora_path = path.join(self.lora_path, 'concepts',
                                              f"{concept_lora_file}.safetensors")
                if path.exists(concept_lora_path):
                    lora_paths.append(concept_lora_path)
                    concept_strength = config.get('concept_strength', 0.33)
                    weights.append(concept_strength)

            self.load_loras(lora_paths, weights)

            image = self.generate(
                ref_image_path=ref_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=ref_image_strength,
                seed=seed,
                height=height,
                width=width)

            if image:
                # Save image
                output_path = self._get_timestamped_path(prompt)
                image.save(output_path)
                logger.info(f"Saved image to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error in edit_from_config: {e}")
            return False

    def _get_timestamped_path(self, prompt: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).replace(" ", "_")
        return self.output_dir / f"{timestamp}_{safe_prompt}.png"


if __name__ == "__main__":
    generator = ImageGenerator()
    generator.load_model()

    while True:
        try:
            input("Press Enter\n")
            generator.generate_from_yaml("/root/.imgenie/prompts/prompt.yaml")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e
