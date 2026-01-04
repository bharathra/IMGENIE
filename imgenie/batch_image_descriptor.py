# module to label multiple images at once

import os
from typing import List
from image_describer import IMGxTXT


class BatchImageDescriptor:

    def __init__(self, folder: str):
        self.folder = folder
        self.image_describer = IMGxTXT()
        self.image_describer.load_model()

    def generate_descriptors(self):
        for filename in os.listdir(self.folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(self.folder, filename)
                img = self.image_describer.get_image_from_path(image_path)
                result = self.image_describer.describe(img)
                # save description to a text file
                self.save_description(filename, result['description'])

    def save_description(self, filename: str, description: str):
        # filename without extension
        filenameonly = os.path.splitext(filename)[0]
        output_file = os.path.join(self.folder, f"{filenameonly}.txt")
        with open(output_file, 'w') as f:
            f.write(description)


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing images: ")
    batch_descriptor = BatchImageDescriptor(folder_path)
    batch_descriptor.generate_descriptors()
    print("Image descriptions generated and saved.")
