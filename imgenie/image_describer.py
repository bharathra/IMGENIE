#!/usr/bin/env python3

import io
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/root/.imgenie/models/fancyfeast.joycaption/weights"


class ImageDescriber:

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 input_dir: str = "/root/.imgenie/input",
                 output_dir: str = "/root/.imgenie/output"):
        # Setup output directory
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

    def load_model(self) -> bool:
        # Load the pipeline
        try:
            load_start = time.time()
            # logger.info(f"Loading model: {self.model}")

            self.i2t_processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.i2t_model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True).to("cuda:0")

            load_time = (time.time() - load_start) * 1000
            logger.info(f"Model loaded in {load_time:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def unload_model(self) -> None:
        if self.i2t_model is not None:
            del self.i2t_model
            torch.cuda.empty_cache()
            logger.info("Model unloaded successfully.")
        else:
            logger.warning("No model to unload.")

    def describe(self, img: PILImage.Image,
                 prompt: Optional[str] = None) -> dict:

        # Each value in "content" has to be a list of dicts with types ("text", "image")
        if prompt is None:
            prompt = "Describe the image in detail. Format the response as a single comprehensive paragraph."

        conversation = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        convo_string = self.i2t_processor.apply_chat_template(conversation,
                                                              tokenize=False,
                                                              add_generation_prompt=True)

        # Process the inputs
        inputs = self.i2t_processor(text=[convo_string],
                                    images=[img],
                                    return_tensors="pt").to('cuda:0')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = self.i2t_model.generate(**inputs,
                                               max_new_tokens=512,
                                               do_sample=True,
                                               suppress_tokens=None,
                                               use_cache=True,
                                               temperature=0.6,
                                               top_k=None,
                                               top_p=0.9)[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = self.i2t_processor.tokenizer.decode(generate_ids,
                                                      skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
        caption = caption.strip()

        return {"description": caption}

    def get_image_from_bytes(self, img_data: bytes) -> PILImage.Image:
        return PILImage.open(io.BytesIO(img_data)).convert("RGB")

    def get_image_from_array(self, img_array: np.ndarray) -> PILImage.Image:
        return PILImage.fromarray(img_array.astype("uint8")).convert("RGB")

    def get_image_from_path(self, img_path: str) -> PILImage.Image:
        return PILImage.open(img_path).convert("RGB")


if __name__ == "__main__":
    # Load model and wait for user prompt.
    server = ImageDescriber()
    server.load_model()

    # # Generate image and save to disk. And wait for next prompt until user exits.
    # while True:
    #     image_path = input("Enter image path (or 'exit' to quit): ")
    #     if image_path.lower() == 'exit':
    #         break
    #     img = server.get_image_from_path(image_path)
    #     result = server.describe(img)
    #     print("Generated Description:", result["description"])

    # Ask for a folder path and describe all images in it
    folder_path = input("Enter folder path containing images (or 'exit' to quit): ")
    if folder_path.lower() != 'exit':
        folder = Path(folder_path)
        for img_file in folder.glob("*.*"):
            try:
                img = server.get_image_from_path(str(img_file))
                result = server.describe(img)
                print(f"Image: {img_file.name} - Description: {result['description']}")
                # Save description to a text file
                desc_file = server.output_dir / f"{img_file.stem}.txt"
                with open(desc_file, 'w') as f:
                    f.write(result['description'])

            except Exception as e:
                logger.error(f"Error processing image {img_file.name}: {e}")
    else:
        print("Exiting...")
        exit()
