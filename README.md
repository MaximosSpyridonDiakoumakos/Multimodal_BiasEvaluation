# MultiModal Bias Evaluation

## Overview
This project's purpose is to test proposed bias evaluation metrics for multimodal machine learning models.
Below are the links for the tested models as found in HuggingFace

This is the bathcelor thesis.

# Quickstart Guide - MBEL
## Requirements
Install dependencies (recomended python version 3.9+):
pip install -r requirements.txt

Dependencies include:
PyTorch (with CUDA if available)
Hugging Face diffusers and transformers
Pillow
NumPy, Matplotlib, Seaborn
Accelerate (for efficient model inference)

## Model Configuration
Model identifiers are defined in model_config.py:

Stable Diffusion (text-to-image):
stabilityai/stable-diffusion-2-1
sd-legacy/stable-diffusion-v1-5
stabilityai/stable-diffusion-xl-base-1.0
CompVis/stable-diffusion-v1-4

BLIP (image-to-text):
Salesforce/blip-image-captioning-base
Salesforce/blip-image-captioning-large

CLIP (evaluation):
openai/clip-vit-base-patch32

## Running Evaluation
The main entry point is main.py.
For example python main.py

This will:
Load model configurations.
Run text-to-image evaluation (t2i.py).
Run image-to-text evaluation (i2t.py).
Compute bias metrics.
Generate visualizations (default enabled).

## Output
Generated images: saved in t2i_outputs/
Metrics and plots: displayed in the console or saved as figures, depending on configuration (plot_data=True/False).
Results dictionary: returned from main() if used as a module.

## Key Functions
run_text_to_image_evaluation(...) – generates images and evaluates them with CLIP.
run_image_to_text_evaluation(...) – captions images and analyzes captions with metrics.
evaluationFunctions/ – contains bias metrics and visualization helpers

## Script Resources

Text to Image:
stabel-diffusion-v1-5:
https://huggingface.co/sd-legacy/stable-diffusion-v1-5
stable-diffusion-xl-base-1.0 (SDXL)
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
stable-diffusion-v1-4 (classic baseline)
https://huggingface.co/CompVis/stable-diffusion-v1-4
stable-diffusion-v2-1
https://huggingface.co/stabilityai/stable-diffusion-2-1

Image to Text:
blip-image-captioning-base
https://huggingface.co/Salesforce/blip-image-captioning-base
blip-image-captioning-large
https://huggingface.co/Salesforce/blip-image-captioning-large
