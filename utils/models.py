controlnet_path = "lllyasviel/ControlNet"
controlnet_model = "lllyasviel/sd-controlnet-hed"
safety_model = "CompVis/stable-diffusion-safety-checker"

base_models = ["runwayml/stable-diffusion-v1-5", "Lykon/dreamshaper-8"]

single_file_base_models = [
    "https://huggingface.co/rehanhaider/sd-loras/impressionismOil_sd15.safetensors",
    "https://huggingface.co/rehanhaider/sd-loras/ghostmix_v20Bakedvae.safetensors",
    "https://huggingface.co/rehanhaider/sd-loras/helloflatart_V12a.safetensors",
]

loras_list = [
    "retro.safetensors",
    "disney.safetensors",
    "MoXinV1.safetensors",
    "CyberPunkAI.safetensors",
    "lord-ak-pop-art-style-v1.safetensors",
    "Watercolor_Painting_by_vizsumit.safetensors",
]
