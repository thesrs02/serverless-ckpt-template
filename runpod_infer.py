# ./runpod_infer.py
import runpod
from src.sd_runner import init, predict
from utils.runpod_validate import validate_job_input

init_dict = init()


def handler(job: dict) -> dict:
    job_input = job["input"]
    validated_input = validate_job_input(job_input)

    if "errors" in validated_input:
        return {"errors": validated_input["errors"]}

    input = validated_input["validated_input"]

    return predict(init_dict, input)


runpod.serverless.start({"handler": handler})
