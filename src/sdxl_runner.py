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
    KDPM2AncestralDiscreteScheduler,
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
    end_time = time.time()

    print(f"setup time: {end_time - start_time}")
    return pipe


def create_canny_image(image_url, max_size=768):
    image = load_image(image_url)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    width, height = image.size
    aspect_ratio = float(width) / float(height)

    if width > height:
        # Landscape
        new_width = min(width, max_size)
        new_height = int(new_width / aspect_ratio)
    else:
        # Portrait
        new_height = min(height, max_size)
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    return resized_image


def predict(input_map, pipe):
    prompt = input_map["prompt"]
    negative_prompt = input_map["negative_prompt"]

    image_url = input_map["image_url"]
    output_width = input_map["output_width"]
    output_height = input_map["output_height"]
    input_image_max_size = input_map["input_image_max_size"]
    controlnet_conditioning_scale = input_map["controlnet_conditioning_scale"]

    seed = input_map["seed"]
    # scheduler = input_map["scheduler"]
    guidance_scale = input_map["guidance_scale"]
    num_inference_steps = input_map["num_inference_steps"]
    canny_image = create_canny_image(image_url, input_image_max_size)
    #
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    optional_params = {}
    if output_width is not None:
        optional_params["width"] = output_width
    if output_height is not None:
        optional_params["height"] = output_height

    output = pipe(
        prompt,
        image=canny_image,
        generator=generator,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        **optional_params,
    )

    return upload_images_to_gcloud(output.images)
