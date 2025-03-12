import os
from PIL import Image
import io

def get_image_dimensions(image_path):
    """Get the dimensions of an image"""
    with Image.open(image_path) as img:
        return img.size

def is_valid_image(file_path):
    """Check if a file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def create_thumbnail(image_path, max_size=(300, 300)):
    """Create a thumbnail of an image"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None 