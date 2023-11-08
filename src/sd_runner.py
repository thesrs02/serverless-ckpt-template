import io
import time
import torch
import base64

#
from compel import Compel
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

#
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
)

from utils.models import (
    safety_model,
    models_cache_dir,
    controlnet_path,
    controlnet_model,
)

models_cache_dir = "diffusers-cache"


def init() -> dict:
    start_time = time.time()
    print("Initializing pipeline...")

    hed = HEDdetector.from_pretrained(
        controlnet_path,
        cache_dir=models_cache_dir,
    )

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        safety_model,
        local_files_only=True,
        torch_dtype=torch.float16,
        cache_dir=models_cache_dir,
    )

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        local_files_only=True,
        torch_dtype=torch.float16,
        cache_dir=models_cache_dir,
    )

    end_time = time.time()
    print(f"init time: {end_time - start_time}")

    return {
        "hed": hed,
        "controlnet": controlnet,
        "safety_checker": safety_checker,
    }


def load_model_into_pipeline(input: dict) -> dict:
    start_time = time.time()
    print("loading models into pipeline...")

    #
    controlnet = input["controlnet"]
    safety_checker = input["safety_checker"]
    #
    model_file_url = input["model_file_url"]
    base_model_name = input["base_model_name"]
    lora_weights_name = input["lora_weights_name"]

    pipe = (
        StableDiffusionControlNetPipeline.from_pretrained(
            base_model_name,
            controlnet=controlnet,
            local_files_only=True,
            torch_dtype=torch.float16,
            cache_dir=models_cache_dir,
        )
        if model_file_url is None
        else StableDiffusionControlNetPipeline.from_single_file(
            model_file_url,
            local_files_only=True,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            cache_dir=models_cache_dir,
            safety_checker=safety_checker,
        )
    )

    if lora_weights_name is not None:
        pipe.load_lora_weights(
            "rehanhaider/sd-loras",
            local_files_only=True,
            cache_dir=models_cache_dir,
            weight_name=lora_weights_name,
        )
    #
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    #
    end_time = time.time()
    print(f"Done. Load time: {end_time - start_time}")

    return {"compel": compel, "pipe": pipe}


def predict(setup: dict, input: dict) -> str:
    pipe = setup["pipe"]
    compel = setup["compel"]

    seed = input["seed"]
    prompt = input["prompt"]
    image_url = input["image_url"]
    lora_scale = input["lora_scale"]
    guess_mode = input["guess_mode"]

    guidance_scale = input["guidance_scale"]
    negative_prompt = input["negative_prompt"]
    num_inference_steps = input["num_inference_steps"]
    controlnet_conditioning_scale = input["controlnet_conditioning_scale"]

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

    buffered = io.BytesIO()
    output.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return "data:image/png;base64," + image_base64
