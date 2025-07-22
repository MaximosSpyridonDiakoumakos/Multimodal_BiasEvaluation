# i2t_bias_caption_analysis.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from evaluationFunctions.eval_func import *
from model_config import *

def run_image_to_text_evaluation(model_name, metrics=None, prompts=None, gender_categories=None, return_raw_data=False, num_images_per_prompt=1):
    """
    Run image-to-text evaluation with specified model and metrics.
    
    Args:
        model_name: The model to use for image captioning
        metrics: Dictionary of metric functions to evaluate
        prompts: List of prompts to evaluate
        gender_categories: List of gender categories for counting
        return_raw_data: If True, return raw captions and gender counts instead of processed metrics
        num_images_per_prompt: Number of images to analyze per prompt (default: 1)
    
    Returns:
        Dictionary of metric results or raw data based on return_raw_data parameter
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the specified model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Use provided prompts and gender categories
    if prompts is None:
        raise ValueError("prompts parameter is required")
    if gender_categories is None:
        raise ValueError("gender_categories parameter is required")
    image_dir = IMAGE_DIR

    # Check if images exist
    missing_images = []
    for i, prompt in enumerate(prompts):
        for img_idx in range(num_images_per_prompt):
            image_filename = f"image_{i}_img_{img_idx}.png"
            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                missing_images.append(image_filename)
    
    if missing_images:
        print(f"Error: Missing images: {missing_images}")
        print("Please run text_to_image.py first to generate the required images.")
        return {}
    
    # Analyze captions
    captions = []
    for i, prompt in enumerate(prompts):
        for img_idx in range(num_images_per_prompt):
            image_filename = f"image_{i}_img_{img_idx}.png"
            image_path = os.path.join(image_dir, image_filename)
            image = Image.open(image_path).convert("RGB")

            # Get model caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)

            # Print results
            print(f"\nPrompt: {prompt} (img {img_idx + 1})")
            print(f"Model: {model_name}")
            print(f"Caption: {caption}")
            
            # Clear GPU cache to prevent memory issues
            if device == "cuda":
                torch.cuda.empty_cache()

    # Calculate gender counts for metrics
    gender_counts = count_gender_words(captions)
    total = sum(gender_counts.values())
    # Avoid division by zero
    model_counts = [gender_counts["man"] / total if total else 0, gender_counts["woman"] / total if total else 0]

    # Return raw data or calculate metrics based on parameter
    if return_raw_data:
        return {"captions": captions, "gender_counts": gender_counts, "model_counts": model_counts}
    elif metrics:
        results = {}
        # Format data once inside the function
        formatted_data = format_data(model_counts)
        for name, metric_func in metrics.items():
            # Check if the metric function expects additional arguments
            import inspect
            sig = inspect.signature(metric_func)
            if len(sig.parameters) > 1:
                # Pass additional arguments if the function expects them
                value = metric_func(formatted_data, prompts=prompts, captions=captions)
            else:
                # Standard metric calling
                value = metric_func(formatted_data)
            results[name] = value
        return results
    else:
        # Return formatted data for external metric calculation
        return format_data(model_counts)

if __name__ == "__main__":
    # Run with default settings when script is run directly
    run_image_to_text_evaluation(BLIP_LARGE_MODEL)
