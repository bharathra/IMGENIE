#!/usr/bin/env python3

import io
import logging
from pathlib import Path
from turtle import mode
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from imgenie.image_generator import TXTxIMG
from imgenie.image_describer import IMGxTXT
from imgenie.model_manager import ModelManager

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
    model_manager: Optional[ModelManager] = None

    def __init__(self):
        self.model_manager = ModelManager()

        models = self.model_manager.list_models()

        if models is None:
            models = {
                "t2i": "/root/.cache/models/hub/models--Tongyi-MAI--Z-Image-Turbo/",
                "i2t": "/root/.cache/huggingface/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/weights"
            }
        t2i_model_id = models.get("t2i")
        self.t2i_model = TXTxIMG(model=t2i_model_id,
                                 output_dir="/root/.imgenie/txt2img")
        i2t_model_id = models.get("i2t")
        self.i2t_model = IMGxTXT(model=i2t_model_id,
                                 output_dir="/root/.imgenie/img2txt")

    def load_models(self, models: Optional[dict] = None):


        # Load T2I model
        t2i_model_id = models.get("t2i")
        if t2i_model_id:
            model_path = self.model_manager.download_model(t2i_model_id)
            self.t2i_model.load_model(model_path=model_path)
            logger.info(f"T2I model loaded from {model_path}")

        # Load I2T model
        i2t_model_id = models.get("i2t")
        if i2t_model_id:
            model_path = self.model_manager.download_model(i2t_model_id)
            self.i2t_model.load_model(model_path=model_path)
            logger.info(f"I2T model loaded from {model_path}")

    @app.on_event("startup")
    async def startup_event(self):
        self.load_models()

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
        if self.model_manager is None:
            raise HTTPException(status_code=503, detail="Multimodal model not loaded")

        model_path = self.model_manager.download_model(model_id)
        return FileResponse(model_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
