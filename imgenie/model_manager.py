#!/usr/bin/env python3

import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages downloading and storing models from Hugging Face Hub."""

    def __init__(self):
        self.hugging_face_cache_folder = "/root/.cache/huggingface/hub"

    def list_models(self):
        # Placeholder for listing models; implementation depends on specific requirements
        logger.info("Listing available models is not implemented yet.")
        pass

    def download_model(self, model_id: str):

        model_folder = Path(self.hugging_face_cache_folder,
                            f"model--{model_id.replace("/", "--")}")
        Path(model_folder).mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {model_id}...")
        logger.info(f"Destination directory: {model_folder}")
        logger.info("This may take a while...")
        snapshot_download(repo_id=model_id,
                          local_dir=model_folder,
                          local_dir_use_symlinks=False,  # Copies files directly
                          revision="main")
        logger.info("Download complete.")
        return model_folder


if __name__ == "__main__":

    # get model id argument
    model_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_id:
        logger.error("Please provide a model ID as an argument.")
        sys.exit(1)

    manager = ModelManager()
    manager.download_model("fancyfeast/llama-joycaption-beta-one-hf-llava")
