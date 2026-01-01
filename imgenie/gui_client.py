#!/usr/bin/env python3

import sys
import os
import requests
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit,
                             QScrollArea, QFrame, QSplitter
                             )
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette, QIcon
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

SERVER_URL = "http://localhost:8000"


class Worker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mode, prompt=None, image_path=None):
        super().__init__()
        self.mode = mode
        self.prompt = prompt
        self.image_path = image_path

    def run(self):
        try:
            if self.mode == "text_only":
                response = requests.post(f"{SERVER_URL}/generate", data={"prompt": self.prompt})
                if response.status_code == 200:
                    temp_path = "temp_output.png"
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
                    self.finished.emit({"image": temp_path, "description": self.prompt})
                else:
                    self.error.emit(f"Server error: {response.text}")

            elif self.mode == "image_only":
                with open(self.image_path, "rb") as f:
                    resp_desc = requests.post(f"{SERVER_URL}/describe", files={"image": f})

                if resp_desc.status_code == 200:
                    description = resp_desc.json()  # ["description"]
                    # Now generate image from this description
                    resp_gen = requests.post(f"{SERVER_URL}/generate", data={"prompt": description})
                    if resp_gen.status_code == 200:
                        temp_path = "temp_output.png"
                        with open(temp_path, "wb") as f:
                            f.write(resp_gen.content)
                        self.finished.emit({"image": temp_path, "description": description})
                    else:
                        self.error.emit(f"Generation error: {resp_gen.text}")
                else:
                    self.error.emit(f"Description error: {resp_desc.text}")

            elif self.mode == "both":
                with open(self.image_path, "rb") as f:
                    resp_desc = requests.post(f"{SERVER_URL}/modify", files={"image": f},
                                              data={"instructions": self.prompt})

                if resp_desc.status_code == 200:
                    description = resp_desc.json()["description"]
                    # Now generate image from this description
                    resp_gen = requests.post(f"{SERVER_URL}/generate", data={"prompt": description})
                    if resp_gen.status_code == 200:
                        temp_path = "temp_output.png"
                        with open(temp_path, "wb") as f:
                            f.write(resp_gen.content)
                        self.finished.emit({"image": temp_path, "description": description})
                    else:
                        self.error.emit(f"Generation error: {resp_gen.text}")
                else:
                    self.error.emit(f"Modification error: {resp_desc.text}")

        except Exception as e:
            self.error.emit(str(e))


class ModernGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMGEN - AI Creative Suite")
        self.setMinimumSize(1200, 800)
        self.selected_image_path = None
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar / Controls
        sidebar = QFrame()
        sidebar.setFixedWidth(350)
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)

        title = QLabel("IMGEN")
        title.setObjectName("appTitle")
        sidebar_layout.addWidget(title)

        # Input Section
        input_label = QLabel("Input Prompt / Instructions")
        input_label.setObjectName("sectionLabel")
        sidebar_layout.addWidget(input_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your creative prompt or modification instructions here...")
        self.prompt_input.setMaximumHeight(150)
        sidebar_layout.addWidget(self.prompt_input)

        # Image Selection
        self.image_btn = QPushButton("Select Reference Image")
        self.image_btn.clicked.connect(self.select_image)
        sidebar_layout.addWidget(self.image_btn)

        self.image_preview_small = QLabel("No image selected")
        self.image_preview_small.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_small.setFixedSize(310, 200)
        self.image_preview_small.setObjectName("previewSmall")
        sidebar_layout.addWidget(self.image_preview_small)

        self.clear_img_btn = QPushButton("Clear Image")
        self.clear_img_btn.setFlat(True)
        self.clear_img_btn.clicked.connect(self.clear_image)
        sidebar_layout.addWidget(self.clear_img_btn)

        sidebar_layout.addStretch()

        # Action Button
        self.generate_btn = QPushButton("GENERATE")
        self.generate_btn.setObjectName("generateBtn")
        self.generate_btn.setFixedHeight(60)
        self.generate_btn.clicked.connect(self.start_generation)
        sidebar_layout.addWidget(self.generate_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        sidebar_layout.addWidget(self.status_label)

        # Main Canvas Area
        canvas_area = QWidget()
        canvas_area.setObjectName("canvasArea")
        canvas_layout = QVBoxLayout(canvas_area)

        self.result_image = QLabel("Generated image will appear here")
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setObjectName("resultImage")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.result_image)
        canvas_layout.addWidget(scroll)

        self.result_description = QTextEdit()
        self.result_description.setReadOnly(True)
        self.result_description.setMaximumHeight(100)
        self.result_description.setPlaceholderText("Model-generated description will appear here...")
        canvas_layout.addWidget(self.result_description)

        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(canvas_area)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f12;
            }
            #sidebar {
                background-color: #1a1a20;
                border-right: 1px solid #2d2d35;
            }
            #canvasArea {
                background-color: #0f0f12;
            }
            #appTitle {
                color: #ffffff;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 10px;
                letter-spacing: 2px;
            }
            #sectionLabel {
                color: #88889a;
                font-size: 14px;
                font-weight: bold;
                text-transform: uppercase;
            }
            QTextEdit {
                background-color: #25252e;
                color: #e0e0e0;
                border: 1px solid #3d3d4a;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #353545;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #454558;
            }
            #generateBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7b61ff, stop:1 #00d1ff);
                color: white;
                font-size: 18px;
                letter-spacing: 2px;
            }
            #generateBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8b75ff, stop:1 #20daff);
            }
            #previewSmall {
                background-color: #121217;
                border: 2px dashed #3d3d4a;
                border-radius: 12px;
                color: #555566;
            }
            #resultImage {
                background-color: #000000;
                color: #333333;
                font-size: 18px;
            }
            #statusLabel {
                color: #666675;
                font-size: 12px;
                font-style: italic;
            }
        """)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_preview_small.setPixmap(pixmap.scaled(self.image_preview_small.size(
            ), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_preview_small.setText("")

    def clear_image(self):
        self.selected_image_path = None
        self.image_preview_small.setPixmap(QPixmap())
        self.image_preview_small.setText("No image selected")

    def start_generation(self):
        prompt = self.prompt_input.toPlainText().strip()
        image = self.selected_image_path

        if not prompt and not image:
            self.status_label.setText("Please provide text or image input.")
            return

        self.generate_btn.setEnabled(False)
        self.status_label.setText("Processing instructions...")

        mode = "text_only"
        if prompt and not image:
            mode = "text_only"
        elif image and not prompt:
            mode = "image_only"
        else:
            mode = "both"

        self.worker = Worker(mode, prompt, image)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, result):
        self.generate_btn.setEnabled(True)
        self.status_label.setText("Generation complete!")

        pixmap = QPixmap(result["image"])
        self.result_image.setPixmap(pixmap.scaled(self.result_image.size(
        ), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.result_image.setText("")

        self.result_description.setText(result["description"])

    def on_error(self, message):
        self.generate_btn.setEnabled(True)
        self.status_label.setText(f"Error: {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ModernGUI()
    window.show()
    sys.exit(app.exec())
