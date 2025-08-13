# prompts_config.py
# Shared configuration for prompts used in both image_to_text.py and text_to_image.py

# Image directory
IMAGE_DIR = "t2i_outputs"

# Model configurations
# BLIP Models - it2
BLIP_BASE_MODEL = "Salesforce/blip-image-captioning-base"
BLIP_LARGE_MODEL = "Salesforce/blip-image-captioning-large"

# Stable Diffusion Models - t2i
STABLE_DIFFUSION_MODEL_v2_1 = "stabilityai/stable-diffusion-2-1"
STABLE_DIFFUSION_MODEL_v1_5 = "sd-legacy/stable-diffusion-v1-5"
STABLE_DIFFUSION_XL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STABLE_DIFFUSION_V1_4_MODEL = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL = "openai/clip-vit-base-patch32" 