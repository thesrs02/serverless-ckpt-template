import cv2
import time
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from utils.gcloud_upload import upload_images_to_gcloud


from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)

MODEL_CACHE_DIR = "diffusers-cache"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROL_NET_MODE_ID = "diffusers/controlnet-canny-sdxl-1.0"


def init():
    start_time = time.time()
    print("Loading pipeline...")

    controlnet = ControlNetModel.from_pretrained(
        CONTROL_NET_MODE_ID,
        local_files_only=True,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16,
    )

    vae = AutoencoderKL.from_pretrained(
        VAE_ID,
        local_files_only=True,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        local_files_only=True,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    end_time = time.time()

    print(f"setup time: {end_time - start_time}")
    return pipe


def create_canny_image(image_url):
    image = load_image(image_url)
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def predict(input_map, pipe):
    prompt = input_map["prompt"]
    negative_prompt = input_map["negative_prompt"]

    image_url = input_map["image_url"]
    controlnet_conditioning_scale = input_map["controlnet_conditioning_scale"]

    guidance_scale = input_map["guidance_scale"]
    num_inference_steps = input_map["num_inference_steps"]
    canny_image = create_canny_image(image_url)

    optional_params = {}
    if negative_prompt is not None:
        optional_params["negative_prompt"] = negative_prompt

    output = pipe(
        prompt,
        image=canny_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        **optional_params,
    )

    return upload_images_to_gcloud(output.images)
