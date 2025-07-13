# i2t_bias_caption_analysis.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from evaluationFunctions.evaluation_functions import *
from prompts_config import *

def run_image_to_text_evaluation(model_name, metrics=None, prompts=None, gender_categories=None, return_raw_data=False):
    """
    Run image-to-text evaluation with specified model and metrics.
    
    Args:
        model_name: The model to use for image captioning
        metrics: Dictionary of metric functions to evaluate
        prompts: List of prompts to evaluate
        gender_categories: List of gender categories for counting
        return_raw_data: If True, return raw captions and gender counts instead of processed metrics
    
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
        image_path = os.path.join(image_dir, f"image_{i}.png")
        if not os.path.exists(image_path):
            missing_images.append(f"image_{i}.png")
    
    if missing_images:
        print(f"Error: Missing images: {missing_images}")
        print("Please run text_to_image.py first to generate the required images.")
        return {}
    
    # Analyze captions
    captions = []
    for i, prompt in enumerate(prompts):
        image_path = os.path.join(image_dir, f"image_{i}.png")
        image = Image.open(image_path).convert("RGB")

        # Get model caption
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        mg = miss_rate(prompt, caption)
        hj = hallucination_score(prompt, caption)
        captions.append(caption)

        # Print results
        print(f"\nPrompt: {prompt}")
        print(f"Model: {model_name}")
        print(f"Caption: {caption}")
        print(f"Miss Rate: {mg} | Hallucination Score: {hj:.2f}\n")
        
        # Clear GPU cache to prevent memory issues
        if device == "cuda":
            torch.cuda.empty_cache()

    # After all prompts, calculate and print real distribution bias
    gender_counts = count_gender_words(captions, gender_categories)
    total = sum(gender_counts.values())
    # Avoid division by zero
    model_counts = [gender_counts["man"] / total if total else 0, gender_counts["woman"] / total if total else 0]
    ideal_counts = [0.5, 0.5]
    db = distribution_bias(model_counts, ideal_counts)
    print("\n=== Distribution Bias (Real Counts) ===")
    print(f"Model: Distribution Bias = {db:.2f} (Counts: {gender_counts})")

    # Return raw data or calculate metrics based on parameter
    if return_raw_data:
        return {"captions": captions, "gender_counts": gender_counts, "model_counts": model_counts}
    elif metrics:
        results = {}
        ideal_distribution = np.array([0.5, 0.5])
        # Format data once inside the function
        formatted_data = format_data(model_counts)
        for name, metric_func in metrics.items():
            value = metric_func(formatted_data, ideal_distribution)
            results[name] = value
        return results
    else:
        # Return formatted data for external metric calculation
        return format_data(model_counts)

if __name__ == "__main__":
    # Run with default settings when script is run directly
    run_image_to_text_evaluation(BLIP_LARGE_MODEL)
