import io
import time
import torch
import base64

from controlnet_aux import HEDdetector
from diffusers.utils import load_image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

models_cache_dir = "diffusers-cache"


def init(local_files_only=True):
    start_time = time.time()
    print("Loading pipeline...")

    hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-hed",
        torch_dtype=torch.float16,
        cache_dir=models_cache_dir,
        local_files_only=local_files_only,
    )

    openjourney = StableDiffusionControlNetPipeline.from_pretrained(
        "prompthero/openjourney-v4",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=models_cache_dir,
        local_files_only=local_files_only,
    )

    dreamshaper = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/dreamshaper-7",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=models_cache_dir,
        local_files_only=local_files_only,
    )

    openjourney.scheduler = UniPCMultistepScheduler.from_config(
        openjourney.scheduler.config
    )
    dreamshaper.scheduler = UniPCMultistepScheduler.from_config(
        dreamshaper.scheduler.config
    )

    openjourney.enable_model_cpu_offload()
    dreamshaper.enable_model_cpu_offload()

    end_time = time.time()
    print(f"setup time: {end_time - start_time}")

    return {"hed": hed, "openjourney": openjourney, "dreamshaper": dreamshaper}


def predict(setup: dict, input: dict) -> str:
    start_time = time.time()
    print("Generating...")

    hed = setup["hed"]
    openjourney = setup["openjourney"]
    dreamshaper = setup["dreamshaper"]

    prompt = input["prompt"]
    image_url = input["image_url"]

    model_name = input["model_name"]

    guidance_scale = input["guidance_scale"]
    negative_prompt = input["negative_prompt"]

    num_inference_steps = input["num_inference_steps"]

    image = load_image(image_url).convert("RGB")
    image = hed(image)

    pipe = openjourney if model_name == "openjourney" else dreamshaper

    images = pipe(
        prompt,
        image=image,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
    ).images

    output = images[0]

    buffered = io.BytesIO()
    output.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    end_time = time.time()
    print(f"Generation time: {end_time - start_time}")

    return "data:image/png;base64," + image_base64
