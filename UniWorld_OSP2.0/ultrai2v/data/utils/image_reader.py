import os
import torch
import numpy as np
from typing import Literal
import typing
from pathlib import Path

import PIL

ImageLayoutType = Literal["HWC", "CHW"]
ImageArrayType = Literal["numpy", "torch", "PIL"]
ImageMaxSize = 100 * 1024 * 1024

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions


class ImageReader:
    """
    Abstract base class defining the common image processing interface
    """
    def __init__(self, image_path: str, layout: ImageLayoutType = "CHW", array_type: ImageArrayType = "torch", **kwargs):
        """
        Initialize image source
        
        Args:
            image_path: String path to image file
            layout (ImageLayoutType): 
                Desired tensor layout format. Options:
                - "CHW": Channel, Height, Width (default)
                - "HWC": Height, Width, Channel
            array_type (imageArrayType):
                Target array container type. Options:
                - "torch": PyTorch tensors (default)
                - "numpy": NumPy ndarrays
                - "PIL": pillow image instance
        """
        self.image_path = Path(image_path)
        self.layout = layout
        self.array_type = array_type
        self._validate_params()

    def _validate_params(self):
        """param validation"""
        if not is_image_file(str(self.image_path)):
            raise ValueError(f"Invalid image type: {self.image_path}")
        if os.path.getsize(str(self.image_path)) > ImageMaxSize:
            raise ValueError(f"The image has to be less than {ImageMaxSize / (1024 * 1024)} M")
        if self.layout not in typing.get_args(ImageLayoutType):
            raise ValueError(f"Invalid image layout type: {self.layout}")
        if self.array_type not in typing.get_args(ImageArrayType):
            raise ValueError(f"Invalid image array type: {self.array_type}")

    def _load_image(self, convert_type="RGB", **kwargs):
        image = PIL.Image.open(str(self.image_path))
        if self.array_type == "PIL":
            return image
        image = image.convert(convert_type)
        image = np.array(image) # H W C or L
        if self.layout == "CHW" and image.ndim == 3:
            image = image.transpose(2, 0, 1)
        if self.array_type == "torch":
            image = torch.from_numpy(image)
        return image
            
    def load_image(self, convert_type="RGB", **kwargs):
        return self._load_image(convert_type, **kwargs)

