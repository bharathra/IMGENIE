#!/usr/bin/env python3

import io
import yaml
import logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from imgenie.image_generator import TXTxIMG
from imgenie.image_describer import IMGxTXT
from huggingface_hub import snapshot_download

from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMGEN Server")

# Constants
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ModelServer:

    # Global model holders
    t2i_model: Optional[TXTxIMG] = None
    i2t_model: Optional[IMGxTXT] = None
    load_status: dict = {"t2i": False, "i2t": False}

    def __init__(self):
        self.hugging_face_cache_folder = "/root/.cache/huggingface/hub"
        # read an YAML config file if exists
        config_path = Path("imgenie.config.default.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        
        # get txt2img and img2txt configs 
        t2i_cfg = config.get("txt2img", {})
        t2i_cfg.update(t2i_cfg)
        i2t_cfg = config.get("img2txt", {})
        i2t_cfg.update(i2t_cfg)

        self.t2i_model = TXTxIMG(model=t2i_cfg.get("model_path", "Tongyi-MAI/Z-Image-Turbo/"),
                                 output_dir=t2i_cfg.get("output_path", "/root/.imgenie/txt2img"))
        self.i2t_model = IMGxTXT(model=i2t_cfg.get("model_path", "/root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/weights"),
                                 output_dir=i2t_cfg.get("output_path", "/root/.imgenie/img2txt"))

    @app.on_event("startup")
    async def startup_event(self):
        pass

    @app.on_event("shutdown")
    async def shutdown_event(self):
        pass

    @app.post("/load_model")
    async def _load_model(self, model_type: str):
        self._load_models(type=[model_type])
        return JSONResponse(content={"status": "models loaded"})

    @app.get("/get_model_load_status")
    async def get_model_load_status(self):
        return JSONResponse(content=self.load_status)

    @app.post("/generate")
    async def generate(self, prompt: str = Form(...)):
        if self.t2i_model is None:
            raise HTTPException(status_code=503, detail="T2I model not loaded")
        path, _ = self.t2i_model.generate(prompt=prompt)
        return FileResponse(path)

    @app.post("/describe")
    async def describe(self, image: UploadFile = File(...)):
        if self.i2t_model is None:
            raise HTTPException(status_code=503, detail="I2T model not loaded")
        img_data = await image.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        result = self.i2t_model.describe(img)
        return result["description"]

    @app.post("/download_model")
    async def download_model(self, model_id: str = Form(...)):
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
        return FileResponse(model_folder)

    def _load_models(self, type: List[str] = ["t2i", "i2t"]):
        for t in type:
            if t not in ["t2i", "i2t"]:
                logger.warning(f"Unknown model type: {t}. Skipping.")
                continue
            logger.info(f"Loading {t} model...")
            if t == "t2i" and self.t2i_model is not None:
                self.t2i_model.load_model()
                self.load_status["t2i"] = True
            elif t == "i2t" and self.i2t_model is not None:
                self.i2t_model.load_model()
                self.load_status["i2t"] = True


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
