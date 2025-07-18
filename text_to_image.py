# t2i_bias_gender_mad.py

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from evaluationFunctions.evaluation_functions import *
from prompts_config import *

from diffusers.pipelines.stable_diffusion import safety_checker

def run_text_to_image_evaluation(model_name, metrics=None, prompts=None, gender_labels=None, return_raw_data=False):
    """
    Run text-to-image evaluation with specified model and metrics.
    
    Args:
        model_name: The model to use for text-to-image generation
        metrics: Dictionary of metric functions to evaluate
        prompts: List of prompts to evaluate
        gender_labels: List of gender labels for CLIP evaluation
        return_raw_data: If True, return raw gender counts instead of processed metrics
    
    Returns:
        Dictionary of metric results or raw data based on return_raw_data parameter
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load T2I model
    t2i_pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)

    # Load CLIP for similarity scoring
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Use provided prompts and gender labels
    if prompts is None:
        raise ValueError("prompts parameter is required")
    if gender_labels is None:
        raise ValueError("gender_labels parameter is required")

    gender_counts = {"man": 0, "woman": 0}
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for i, prompt in enumerate(prompts):
        # Generate image
        image = t2i_pipe(prompt).images[0]
        
        # Save image and get full path
        image_path = os.path.join(IMAGE_DIR, f"image_{i}.png")
        image.save(image_path)
        
        # Print where the image is stored
        full_path = os.path.abspath(image_path)
        print(f"Image saved at: {full_path}")

        # Evaluate with CLIP
        inputs = clip_processor(text=gender_labels, images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image.softmax(dim=1)
        probs = logits_per_image[0].tolist()
        
        gender = "man" if probs[0] > probs[1] else "woman"
        gender_counts[gender] += 1
        print(f"{prompt} => {gender} ({probs})")
        
        # Clear GPU cache to prevent memory issues
        if device == "cuda":
            torch.cuda.empty_cache()

    # Return raw data or calculate metrics based on parameter
    if return_raw_data:
        return gender_counts
    elif metrics:
        results = {}
        # Format data once inside the function
        formatted_data = format_data(gender_counts)
        for name, metric_func in metrics.items():
            value = metric_func(formatted_data)
            results[name] = value
        return results
    else:
        # Return formatted data for external metric calculation
        print("\nGender Distribution:", gender_counts)
        return format_data(gender_counts)

if __name__ == "__main__":
    # Run with default settings when script is run directly
    run_text_to_image_evaluation(STABLE_DIFFUSION_MODEL_v1_5)
