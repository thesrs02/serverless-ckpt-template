import time
import torch
from utils.canny import load_canny_image
from utils.image_utils import resize_large_images
from utils.gcloud_upload import upload_images_to_gcloud
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)

MODEL_CACHE_DIR = "diffusers-cache"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROL_NET_MODE_ID = "diffusers/controlnet-canny-sdxl-1.0"


def init():
    start_time = time.time()
    print("Loading pipeline...")

    controlnet = ControlNetModel.from_pretrained(
        CONTROL_NET_MODE_ID,
        local_files_only=True,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    )

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        local_files_only=True,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    )

    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_ID,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    end_time = time.time()

    print(f"setup time: {end_time - start_time}")
    return {"pipe": pipe, "refiner": refiner}


def predict(input_map, setup):
    pipe = setup["pipe"]
    refiner = setup["refiner"]

    image_url = input_map["image_url"]

    prompt = input_map["prompt"]
    prompt_2 = input_map["prompt_2"]

    negative_prompt = input_map["negative_prompt"]
    negative_prompt_2 = input_map["negative_prompt_2"]

    guidance_scale = input_map["guidance_scale"]
    num_inference_steps = input_map["num_inference_steps"]
    controlnet_conditioning_scale = input_map["controlnet_conditioning_scale"]

    refiner_guidance_scale = input_map["refiner_guidance_scale"]
    refiner_num_inference_steps = input_map["refiner_num_inference_steps"]

    resize_output_to = input_map["resize_output_to"]
    canny_min_threshold = input_map["canny_min_threshold"]
    canny_max_threshold = input_map["canny_max_threshold"]
    apply_input_image_enhancers = input_map["apply_input_image_enhancers"]

    canny_image = load_canny_image(
        image_url, apply_input_image_enhancers, canny_min_threshold, canny_max_threshold
    )

    pre_refiner_image = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        image=canny_image,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]

    refined_image = refiner(
        prompt,
        image=pre_refiner_image,
        guidance_scale=refiner_guidance_scale,
        num_inference_steps=refiner_num_inference_steps,
    ).images[0]

    output = resize_large_images(refined_image, resize_output_to)

    return upload_images_to_gcloud([output])
