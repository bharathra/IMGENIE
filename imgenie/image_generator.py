#!/usr/bin/env python3

import logging
from os import path
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import yaml as yf   # to avoid conflict with PyYAML

import numpy as np
import torch
from PIL import Image
from diffusers import (
    ZImageImg2ImgPipeline,
    ZImagePipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel
)
try:
    from diffusers import Krea2Pipeline
except ImportError:
    Krea2Pipeline = None
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
                # Load from directory
                logger.info("Loading model from directory structure")
                
                # Check model_index.json to dynamically load the right pipeline
                import json
                model_index_path = Path(self.model_path) / "model_index.json"
                pipeline_class = ZImageImg2ImgPipeline
                
                if model_index_path.exists():
                    class_name = None
                    try:
                        with open(model_index_path, 'r') as f:
                            model_index = json.load(f)
                        class_name = model_index.get("_class_name")
                        logger.info(f"Detected pipeline class from model_index.json: {class_name}")
                    except Exception as e:
                        logger.warning(f"Error reading model_index.json: {e}. Defaulting to ZImageImg2ImgPipeline.")
                    
                    if class_name is not None:
                        if class_name == "Krea2Pipeline":
                            if Krea2Pipeline is None:
                                raise ImportError(
                                    "Krea2Pipeline is not available in the installed diffusers package. "
                                    "Please install diffusers from source: pip install git+https://github.com/huggingface/diffusers.git"
                                )
                            pipeline_class = Krea2Pipeline
                        elif class_name == "StableDiffusionPipeline":
                            pipeline_class = StableDiffusionPipeline
                        elif class_name == "ZImageImg2ImgPipeline":
                            pipeline_class = ZImageImg2ImgPipeline
                        elif class_name == "ZImagePipeline":
                            pipeline_class = ZImagePipeline
                
                logger.info(f"Using pipeline class: {pipeline_class.__name__}")
                
                if pipeline_class == Krea2Pipeline:
                    self.pipeline = Krea2Pipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        local_files_only=True
                    ).to("cuda:0")
                    
                    # 1. Cast VAE to float16 to optimize decode speed
                    if hasattr(self.pipeline, "vae") and self.pipeline.vae is not None:
                        self.pipeline.vae.to(dtype=torch.float16)
                        logger.info("Casted Krea2 VAE to float16 to optimize decode speed")
                    
                    # 2. Patch text encoding to use padding='longest' to avoid padding tokens
                    original_get_text_hidden_states = self.pipeline.get_text_hidden_states
                    def patched_get_text_hidden_states(pipeline_self, prompt, max_sequence_length=512, device=None):
                        device = device or pipeline_self._execution_device
                        prompt = [prompt] if isinstance(prompt, str) else prompt
                        prefix_idx = pipeline_self.prompt_template_encode_start_idx
                        text = [pipeline_self.prompt_template_encode_prefix + e for e in prompt]
                        
                        text_tokens = pipeline_self.tokenizer(
                            text,
                            truncation=True,
                            padding="longest",
                            max_length=max_sequence_length + prefix_idx - pipeline_self.prompt_template_encode_num_suffix_tokens,
                            return_tensors="pt",
                        ).to(device)
                        suffix_tokens = pipeline_self.tokenizer([pipeline_self.prompt_template_encode_suffix] * len(text), return_tensors="pt").to(device)

                        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
                        attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

                        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                        outputs = pipeline_self.text_encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            output_hidden_states=True,
                        )
                        hidden_states = torch.stack([outputs.hidden_states[i] for i in pipeline_self.text_encoder_select_layers], dim=2)

                        hidden_states = hidden_states[:, prefix_idx:]
                        attention_mask = attention_mask[:, prefix_idx:]
                        
                        return hidden_states, attention_mask

                    self.pipeline.get_text_hidden_states = patched_get_text_hidden_states.__get__(self.pipeline, self.pipeline.__class__)
                    logger.info("Patched Krea2Pipeline.get_text_hidden_states to use padding='longest'")

                    # 3. Patch transformer forward pass to dynamically handle position_ids and bypass attention mask when it's all-True (which enables FlashAttention)
                    original_forward = self.pipeline.transformer.forward
                    def patched_transformer_forward(transformer_self, hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None, **kwargs):
                        text_seq_len = encoder_hidden_states.shape[1]
                        image_seq_len = hidden_states.shape[1]
                        grid_size = int(image_seq_len ** 0.5)
                        
                        device = hidden_states.device
                        text_ids = torch.zeros(text_seq_len, 3, dtype=position_ids.dtype, device=device)
                        image_ids = torch.zeros(grid_size, grid_size, 3, dtype=position_ids.dtype, device=device)
                        image_ids[..., 1] = torch.arange(grid_size, device=device)[:, None]
                        image_ids[..., 2] = torch.arange(grid_size, device=device)[None, :]
                        image_ids = image_ids.reshape(grid_size * grid_size, 3)
                        dynamic_position_ids = torch.cat([text_ids, image_ids], dim=0)

                        if encoder_attention_mask is not None and encoder_attention_mask.all():
                            encoder_attention_mask = None

                        return original_forward(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timestep,
                            position_ids=dynamic_position_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            **kwargs
                        )

                    self.pipeline.transformer.forward = patched_transformer_forward.__get__(self.pipeline.transformer, self.pipeline.transformer.__class__)
                    logger.info("Patched Krea2Transformer2DModel.forward to enable dynamic position_ids and FlashAttention")
                else:
                    self.pipeline = ZImageImg2ImgPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True).to("cuda:0")
                self.is_single_file_model = False
            
            # # Enable VAE tiling to prevent VRAM overflow and system RAM swapping during the final decode step (common for large resolutions like Krea2)
            # if hasattr(self.pipeline, 'enable_vae_tiling'):
            #     try:
            #         self.pipeline.enable_vae_tiling()
            #         logger.info("VAE tiling enabled to optimize memory usage and prevent swapping during decode.")
            #     except Exception as e:
            #         logger.warning(f"Could not enable VAE tiling: {e}")

            # # Explicitly enable benchmarking for optimal convolution performance (this causes the first run of a new resolution to be slower as it profiles algorithms, but subsequent runs are faster)
            # if torch.cuda.is_available():
            #     torch.backends.cudnn.benchmark = True

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

    def load_loras(self, loras: list, weights: Optional[list] = None) -> dict:
        """Loads multiple LoRA adapters into the pipeline correctly from local files.
        
        Returns:
            dict: {'failed_loras': list of lora paths that failed to load}
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline is not loaded.")

        # Restore original txtfusion projector weight if previously modified
        if getattr(self, '_orig_txtfusion_projector_weight', None) is not None:
            if hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'text_fusion'):
                self.pipeline.transformer.text_fusion.projector.weight.data.copy_(self._orig_txtfusion_projector_weight)

        # First, unload existing loras
        # Always try to clear any adapters just in case to prevent adapter name collision
        try:
            self.pipeline.unload_lora_weights()
        except Exception:
            pass
        self._active_loras = []

        if not loras:
            return {'failed_loras': []}

        # load added loras
        valid_indices = []
        failed_loras = []
        peft_adapters = []
        peft_weights = []

        for i, path_str in enumerate(loras):
            path = Path(path_str)
            adapter_name = f"adapter_{i}"
            
            try:
                # Load state dict to check and potentially convert keys
                state_dict = load_file(str(path))
                
                # If LoRA was trained on base model using Kohya format, it uses 'lora_down'/'lora_up' 
                # and 'transformer.' prefix which diffusers ignores for ZImage models.
                # We convert these to standard PEFT format ('lora_A'/'lora_B' and 'diffusion_model.')
                is_kohya = any('lora_down' in k for k in state_dict.keys())
                if is_kohya:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_key = k
                        if 'lora_down' in new_key:
                            new_key = new_key.replace('lora_down', 'lora_A')
                        elif 'lora_up' in new_key:
                            new_key = new_key.replace('lora_up', 'lora_B')
                            
                        if new_key.startswith('transformer.'):
                            new_key = new_key.replace('transformer.', 'diffusion_model.', 1)
                            
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict

                # Handle non-PEFT / .diff keys (e.g. txtfusion.projector.diff in Krea2 LoRAs)
                adapter_weight = weights[i] if (weights is not None and i < len(weights)) else 1.0
                diff_keys = [k for k in list(state_dict.keys()) if '.diff' in k or k.endswith('.diff')]
                for k in diff_keys:
                    val = state_dict.pop(k)
                    if 'txtfusion.projector' in k and hasattr(self.pipeline, 'transformer') and hasattr(self.pipeline.transformer, 'text_fusion'):
                        if getattr(self, '_orig_txtfusion_projector_weight', None) is None:
                            self._orig_txtfusion_projector_weight = self.pipeline.transformer.text_fusion.projector.weight.data.clone()
                        target_weight = self.pipeline.transformer.text_fusion.projector.weight
                        target_weight.data += (val.to(target_weight.device, dtype=target_weight.dtype) * adapter_weight)
                        logger.info(f"Applied {k} to transformer.text_fusion.projector.weight (scale={adapter_weight})")
                    else:
                        logger.warning(f"Unhandled diff key {k} in LoRA {path_str}")

                # Check for .alpha scalar keys (e.g. in some Krea2 LoRAs) which cause diffusers
                # Krea2 converter to raise ValueError ('state_dict should be empty at this point').
                # Extract the intended alpha/rank scaling factor before stripping .alpha keys so
                # that PEFT loads the adapter at the correct mathematical scale (otherwise PEFT defaults
                # alpha=rank, which can over-scale weights by up to 16x and produce noise).
                alpha_keys = [k for k in state_dict.keys() if k.endswith('.alpha')]
                if alpha_keys:
                    try:
                        sample_alpha = float(state_dict[alpha_keys[0]])
                        rank_keys = [k for k in state_dict.keys() if 'lora_A' in k or 'lora_down' in k]
                        if rank_keys:
                            rank = state_dict[rank_keys[0]].shape[0]
                            if rank > 0:
                                alpha_scale = sample_alpha / float(rank)
                                adapter_weight = adapter_weight * alpha_scale
                                logger.info(f"Detected LoRA alpha={sample_alpha}, rank={rank}. Adjusted adapter scale factor to {alpha_scale:.6f}")
                    except Exception as err:
                        logger.warning(f"Error calculating alpha scale factor: {err}")
                    
                    state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.alpha')}

                if len(state_dict) > 0:
                    # For ZImage models, use prefix=None to avoid key mismatch warnings
                    prefix = None if isinstance(self.pipeline, (ZImageImg2ImgPipeline, ZImagePipeline)) else None
                    
                    self.pipeline.load_lora_weights(
                        state_dict,
                        adapter_name=adapter_name,
                        **({'prefix': prefix} if prefix is not None else {})
                    )
                    peft_adapters.append(adapter_name)
                    if weights is not None and i < len(weights):
                        peft_weights.append(weights[i])

                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error loading LoRA {path_str}: {e}")
                failed_loras.append(path_str)

        # Update active loras to only include successfully loaded ones
        self._active_loras = [loras[i] for i in valid_indices]
        logger.info(f"Active LoRAs after update: {self._active_loras}")
        
        # Set adapters only for registered PEFT adapters
        if peft_adapters:
            try:
                self.pipeline.set_adapters(peft_adapters, adapter_weights=peft_weights if weights is not None else None)
            except Exception as e:
                logger.error(f"Error setting adapters: {e}")
        
        logger.info(f"Successfully loaded LoRAs: {self._active_loras}")
        if failed_loras:
            logger.warning(f"Failed to load LoRAs: {failed_loras}")
        
        return {'failed_loras': failed_loras}

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

    def _is_blank_or_invalid_image(self, image: Image.Image) -> bool:
        try:
            arr = np.asarray(image)
            if arr.size == 0:
                return True
            if np.isnan(arr).any() or np.isinf(arr).any():
                return True
            if np.all(arr == arr.flat[0]):
                return True
            return False
        except Exception as e:
            logger.warning(f"Unable to validate generated image: {e}")
            return False

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

            reference_img = None
            if ref_image_path and path.exists(ref_image_path):
                reference_img = self._load_reference_image(ref_image_path)
                height = reference_img.height
                width = reference_img.width

            def run_generation():
                with torch.no_grad():
                    if reference_img is not None:
                        if isinstance(self.pipeline, (ZImageImg2ImgPipeline, ZImagePipeline)):
                            return self.pipeline(
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
                            ).images[0]
                        elif Krea2Pipeline is not None and isinstance(self.pipeline, Krea2Pipeline):
                            logger.warning("Krea2Pipeline does not natively support image-to-image (ref_image_path). Falling back to text-to-image.")
                            return self.pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                height=height,
                                width=width,
                                generator=generator,
                                callback_on_step_end=callback
                            ).images[0]
                        else:
                            img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipeline.components)
                            return img2img_pipe(
                                prompt=prompt,
                                image=reference_img,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                num_images_per_prompt=1,
                                strength=strength,
                                generator=generator,
                                callback_on_step_end=callback
                            ).images[0]
                    else:
                        if isinstance(self.pipeline, (ZImageImg2ImgPipeline, ZImagePipeline)):
                            return ZImagePipeline(**self.pipeline.components)(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                num_images_per_prompt=1,
                                height=height,
                                width=width,
                                generator=generator,
                                callback_on_step_end=callback
                            ).images[0]
                        elif Krea2Pipeline is not None and isinstance(self.pipeline, Krea2Pipeline):
                            return self.pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                height=height,
                                width=width,
                                generator=generator,
                                callback_on_step_end=callback
                            ).images[0]
                        else:
                            return self.pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                num_images_per_prompt=1,
                                height=height,
                                width=width,
                                generator=generator,
                                callback_on_step_end=callback
                            ).images[0]

            image = run_generation()
            if self._is_blank_or_invalid_image(image):
                logger.warning("First generation produced an invalid or blank image; retrying once.")
                image = run_generation()

            return image

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
