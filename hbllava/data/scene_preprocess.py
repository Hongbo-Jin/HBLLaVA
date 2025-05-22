import os

from PIL import Image, ImageFile
import torch
import ast

from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ScenePreprocess:
    def __init__(self, image_processor):
        self.image_processor = image_processor
    
    def __call__(self, image):
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        
        return image

