import runpod
import src.sdxl_runner as model
from utils.runpod_validate import validate_job_input


base_model = model.init()


def handler(job):
    job_input = job["input"]
    validated_input = validate_job_input(job_input)

    if "errors" in validated_input:
        return {"errors": validated_input["errors"]}

    valid_input = validated_input["validated_input"]
    images = model.predict(valid_input, base_model)

    return images[0]


runpod.serverless.start({"handler": handler})
