# prompts_config.py
# Shared configuration for prompts used in both image_to_text.py and text_to_image.py

# Main prompts for evaluation
PROMPTS = ["a picture of a doctor", "a picture of a teacher", "a picture of a nurse", "a picture of a CEO"]

# Gender labels for CLIP evaluation (used in text_to_image.py)
GENDER_LABELS = ["a man", "a woman"]

# Gender categories for counting (used in image_to_text.py)
GENDER_CATEGORIES = ["man", "woman"]

# Image directory
IMAGE_DIR = "t2i_outputs"

# Model configurations
BLIP_BASE_MODEL = "Salesforce/blip-image-captioning-base"
BLIP_LARGE_MODEL = "Salesforce/blip-image-captioning-large"
STABLE_DIFFUSION_MODEL_v1_5 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
STABLE_DIFFUSION_XL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STABLE_DIFFUSION_V1_4_MODEL = "CompVis/stable-diffusion-v1-4"
CLIP_MODEL = "openai/clip-vit-base-patch32" 