import time
import torch

#
from PIL import Image
from compel import Compel
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    EulerAncestralDiscreteScheduler,
)

MODEL_CACHE_DIR = "diffusers-cache"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROL_NET_MODE_ID = "lllyasviel/ControlNet"


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
    compel = setup["compel"]

    seed = input_map["seed"]
    prompt = input_map["prompt"]
    image_url = input_map["image_url"]
    lora_scale = input_map["lora_scale"]
    guess_mode = input_map["guess_mode"]

    guidance_scale = input_map["guidance_scale"]
    negative_prompt = input_map["negative_prompt"]
    num_inference_steps = input_map["num_inference_steps"]
    controlnet_conditioning_scale = input_map["controlnet_conditioning_scale"]

    #
    image = load_image(image_url)
    prompt_embeds = compel([prompt])
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipe(
        image=image,
        generator=generator,
        guess_mode=guess_mode,
        prompt_embeds=prompt_embeds,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": lora_scale},
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

    output = images[0]

    return output  # in base64
