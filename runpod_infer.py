import runpod
import src.sd_runner as model
from utils.runpod_validate import validate_job_input

model_cache = {}
hed, controlnet, safety_checker = model.init()


def handler(job: dict) -> dict:
    job_input = job["input"]
    validated_input = validate_job_input(job_input)

    if "errors" in validated_input:
        return {"errors": validated_input["errors"]}

    input = validated_input["validated_input"]
    #
    #
    model_file_url = input["model_file_url"]
    base_model_name = input["base_model_name"]
    lora_weights_name = input["lora_weights_name"]
    #
    #
    load_model_dict = {
        "controlnet": controlnet,
        "safety_checker": safety_checker,
        "model_file_url": model_file_url,
        "base_model_name": base_model_name,
        "lora_weights_name": lora_weights_name,
    }

    cache_key = (model_file_url, base_model_name, lora_weights_name)
    if cache_key not in model_cache:
        model_cache[cache_key] = model.load_model_into_pipeline(load_model_dict)

    pipe, compel = model_cache[cache_key]

    #
    #
    image_url = input["image_url"]
    image = hed(image_url)

    return model.predict({"pipe": pipe, "compel": compel}, {**input, "image": image})


runpod.serverless.start({"handler": handler})
