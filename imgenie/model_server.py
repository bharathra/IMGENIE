#!/usr/bin/env python3

import io
import yaml
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from image_generator import ImageGenerator
from image_describer import ImageDescriber
from model_downloader import ModelDownloader

from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMGEN Server")


class ModelType(Enum):
    T2I = "t2i"
    I2T = "i2t"


class ModelServer:

    # Global model holders
    t2i_model: Optional[ImageGenerator] = None
    i2t_model: Optional[ImageDescriber] = None
    load_status: Dict = {ModelType.T2I: False,
                         ModelType.I2T: False}

    def __init__(self, config_path: Optional[str] = "config/imgenie.config.default.yaml"):
        self.model_downloader = ModelDownloader()

        # read an YAML config file if exists
        if config_path and Path(config_path).is_file():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {config_path}. Using default settings.")
            config = {}

        self.root_dir = Path(config.get("root_dir", "/root/.imgenie")).expanduser()
        self.input_folder = self.root_dir.joinpath(Path(config.get("input_path", "input")).expanduser())
        self.output_folder = self.root_dir.joinpath(Path(config.get("output_path", "output")).expanduser())
        self.lora_path = self.root_dir.joinpath(Path(config.get("lora_path", "loras")).expanduser())

        # get txt2img config
        self.t2i_cfg = config.get("txt2img", {})
        self.t2i_model_path = self.t2i_cfg.get("model_path", "")
        self.t2i_prompts_yaml = self.t2i_cfg.get("prompts_yaml", "")
        self.t2i_model = ImageGenerator(model_path=self.t2i_model_path,
                                        input_dir=str(self.input_folder),
                                        output_dir=str(self.output_folder),
                                        lora_path=str(self.lora_path))
        # get img2txt config
        self.i2t_cfg = config.get("img2txt", {})
        self.i2t_model_path = self.i2t_cfg.get("model_path", "")
        self.i2t_model = ImageDescriber(model_path=self.i2t_model_path,
                                        input_dir=str(self.input_folder),
                                        output_dir=str(self.output_folder))

    @app.post("/download_model")
    async def download_model(self, model_id: str = Form(...)) -> FileResponse:
        model_folder = self.model_downloader.download_model(model_id=model_id)
        return FileResponse(model_folder)

    @app.post("/load_model")
    async def load_model(self, model_path: str = Form(...), model_type: str = Form(...)):
        if model_type not in ModelType._value2member_map_:
            raise HTTPException(status_code=400, detail="Invalid model type. Must be 't2i' or 'i2t'.")
        self._load_model(model_path=model_path, model_type=ModelType(model_type))
        return JSONResponse(content={"status": f"Model loaded"})

    @app.post("/unload_model")
    async def unload_model(self, model_type: str = Form(...)):
        if model_type not in ModelType._value2member_map_:
            raise HTTPException(status_code=400, detail="Invalid model type. Must be 't2i' or 'i2t'.")
        self._unload_model(model_type=ModelType(model_type))
        return JSONResponse(content={"status": "Model unloaded"})

    @app.get("/get_model_load_status")
    async def get_model_load_status(self):
        return JSONResponse(content=self.load_status)

    @app.post("/generate")
    async def generate(self):
        if self.t2i_model is None:
            raise HTTPException(status_code=503, detail="T2I model not loaded")
        result = self.t2i_model.generate_from_yaml(yaml_path=self.t2i_prompts_yaml)
        if result is None:
            raise HTTPException(status_code=500, detail="Image generation failed")
        # support either a single path or a (path, meta) tuple/list
        if isinstance(result, (list, tuple)):
            if len(result) == 0:
                raise HTTPException(status_code=500, detail="Image generation returned empty result")
            path = result[0]
        else:
            path = result
        return FileResponse(path)

    @app.post("/describe")
    async def describe(self, image: UploadFile = File(...)):
        if self.i2t_model is None:
            raise HTTPException(status_code=503, detail="I2T model not loaded")
        img_data = await image.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        result = self.i2t_model.describe(img)
        return result["description"]

    def _load_model(self, model_path: str, model_type: ModelType):
        logger.info(f"Loading {model_type} model: {model_path}...")
        if model_type == ModelType.T2I and self.t2i_model is not None:
            self.load_status[ModelType.T2I] = self.t2i_model.load_model()
        elif model_type == ModelType.I2T and self.i2t_model is not None:
            self.load_status[ModelType.I2T] = self.i2t_model.load_model()

    def _unload_model(self, model_type: ModelType):
        logger.info(f"Unloading {model_type} model...")
        if model_type == ModelType.T2I and self.t2i_model is not None:
            self.t2i_model.unload_model()
            self.load_status[ModelType.T2I] = False
        elif model_type == ModelType.I2T and self.i2t_model is not None:
            self.i2t_model.unload_model()
            self.load_status[ModelType.I2T] = False


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    server = ModelServer()
