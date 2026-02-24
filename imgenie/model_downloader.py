#!/usr/bin/env python3

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDownloader:

    def __init__(self, cache_folder: str = "/root/.imgenie/models"):
        self.cache_folder = cache_folder

    def download_model(self, model_id: str) -> Path:
        model_folder = Path(self.cache_folder,
                            f"model--{model_id.replace("/", ".")}")
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
    server = ModelDownloader()
    server.download_model("stabilityai/stable-diffusion-x4-upscaler")
