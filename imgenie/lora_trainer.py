#!/usr/bin/env python3
"""Minimal LoRA trainer for ZImageTurbo.

This module prepares a dataset from folder-labeled image folders and optionally
launches a training command.

Folder structure example:

  /root/.imgenie/training/
    red_car/
      img1.jpg
      img2.png
    golden_retriever/
      dog1.jpg
      dog2.jpg

Every image will be copied into a training dataset folder with a same-name
.txt caption file containing the folder label.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LoraTrainer:
    def __init__(self,
                 training_dir: str = "/root/.imgenie/training",
                 output_dir: str = "/root/.imgenie/loras",
                 model_path: str = "/root/.imgenie/models/TongyiMAI.ZImageTurbo",
                 dataset_dir: Optional[str] = None,
                 train_script: Optional[str] = None):
        self.training_dir = Path(training_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.output_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train_script = Path(train_script) if train_script else Path(__file__).parent / "train_lora.py"

    def _image_files(self, folder: Path) -> List[Path]:
        return [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

    def prepare_concept(self, concept_folder: Path) -> Path:
        concept_name = concept_folder.name
        target_folder = self.dataset_dir / concept_name
        target_folder.mkdir(parents=True, exist_ok=True)

        images = self._image_files(concept_folder)
        if not images:
            logger.warning("Skipping %s because it contains no supported image files.", concept_folder)
            return target_folder

        for image_path in images:
            dst_image = target_folder / image_path.name
            shutil.copy(image_path, dst_image)
            caption_path = target_folder / f"{image_path.stem}.txt"
            caption_path.write_text(concept_name, encoding="utf-8")
            logger.debug("Prepared %s and %s", dst_image, caption_path)

        logger.info("Prepared dataset for concept '%s' with %d images.", concept_name, len(images))
        return target_folder

    def prepare_all(self) -> List[Path]:
        if not self.training_dir.exists():
            raise FileNotFoundError(f"Training data directory does not exist: {self.training_dir}")

        prepared_dirs: List[Path] = []
        for concept_folder in sorted(self.training_dir.iterdir()):
            if concept_folder.is_dir():
                prepared_dirs.append(self.prepare_concept(concept_folder))
        return prepared_dirs

    def build_train_command(self,
                            concept_name: str,
                            dataset_path: Path,
                            lora_output_path: Path,
                            num_epochs: int = 5,
                            batch_size: int = 1,
                            learning_rate: str = "1e-4") -> List[str]:
        if not self.train_script.exists():
            raise ValueError(f"Training script not found: {self.train_script}")

        return [
            "accelerate", "launch", str(self.train_script),
            "--dataset_base_path", str(dataset_path),
            "--output_path", str(lora_output_path),
            "--model_path", str(self.model_path),
            "--concept", concept_name,
            "--num_epochs", str(num_epochs),
            "--train_batch_size", str(batch_size),
            "--learning_rate", learning_rate,
        ]

    def train_concept(self,
                      concept_name: str,
                      dataset_path: Path,
                      num_epochs: int = 5,
                      batch_size: int = 1,
                      learning_rate: str = "1e-4") -> None:
        lora_output_path = self.output_dir / f"lora_{concept_name}"
        lora_output_path.mkdir(parents=True, exist_ok=True)

        cmd = self.build_train_command(
            concept_name=concept_name,
            dataset_path=dataset_path,
            lora_output_path=lora_output_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        logger.info("Starting training for '%s' using %s", concept_name, self.train_script)
        logger.info("Training command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Finished training for '%s'. Output: %s", concept_name, lora_output_path)

    def train_all(self,
                  num_epochs: int = 5,
                  batch_size: int = 1,
                  learning_rate: str = "1e-4") -> None:
        prepared = self.prepare_all()
        if not self.train_script.exists():
            raise ValueError(f"Training script not found: {self.train_script}")

        for dataset_path in prepared:
            concept_name = dataset_path.name
            self.train_concept(concept_name, dataset_path, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ZImageTurbo LoRA training helper.")
    parser.add_argument("--training-dir", default="/root/.imgenie/training", help="Root folder with labeled image subfolders.")
    parser.add_argument("--output-dir", default="/root/.imgenie/loras", help="Output root for datasets and LoRA checkpoints.")
    parser.add_argument("--model-path", default="/root/.imgenie/models/TongyiMAI.ZImageTurbo", help="Path to the ZImageTurbo base model.")
    parser.add_argument("--train-script", help="Training script path to launch with accelerate. Defaults to train_lora.py in the same directory.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--learning-rate", default="1e-4", help="Learning rate for training.")
    parser.add_argument("--concept", help="Optional single concept folder to train. If omitted, all concepts are trained.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = LoraTrainer(
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        train_script=args.train_script,
    )

    if args.concept:
        concept_folder = Path(args.training_dir) / args.concept
        if not concept_folder.exists() or not concept_folder.is_dir():
            raise FileNotFoundError(f"Concept folder not found: {concept_folder}")
        dataset_path = trainer.prepare_concept(concept_folder)
        trainer.train_concept(args.concept, dataset_path, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    else:
        trainer.train_all(num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
