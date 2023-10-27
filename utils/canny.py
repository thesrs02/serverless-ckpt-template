import cv2
import numpy as np
from PIL import Image

from .image_utils import resize_large_images, crop_to_target, sharpen_image
from diffusers.utils import load_image


def load_canny_image(url, enchance_input_image=True, min=100, max=200):
    image = load_image(url)
    image = resize_large_images(image)

    if enchance_input_image:
        image = sharpen_image(image)

    image = crop_to_target(image)
    image = np.array(image)
    image = cv2.Canny(image, min, max)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image
