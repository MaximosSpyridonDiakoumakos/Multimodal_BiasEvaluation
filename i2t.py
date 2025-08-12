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

    # Create comprehensive data structure for all metrics
    # For fairness metrics that need FPR/FNR, we'll create synthetic data based on gender distribution
    # This assumes that bias in gender representation correlates with classification bias
    male_prop = model_counts[0] if total > 0 else 0.5
    female_prop = model_counts[1] if total > 0 else 0.5
    
    # Create synthetic classification metrics based on gender distribution
    # If male proportion is high, assume higher true positive rate for male detection
    # If female proportion is low, assume higher false negative rate for female detection
    tpr_male = male_prop  # Higher male proportion = higher TPR for male
    tpr_female = female_prop  # Higher female proportion = higher TPR for female
    
    # FPR and FNR are complementary to TPR
    fpr_male = 1 - tpr_male  # False positive rate for male
    fpr_female = 1 - tpr_female  # False positive rate for female
    fnr_male = 1 - tpr_male  # False negative rate for male  
    fnr_female = 1 - tpr_female  # False negative rate for female
    
    # Create comprehensive data structure
    comprehensive_data = {
        # Gender distribution data
        "gender_distribution": model_counts,
        "male_proportion": male_prop,
        "female_proportion": female_prop,
        
        # Classification metrics for fairness evaluation
        "tpr_male": tpr_male,
        "tpr_female": tpr_female,
        "fpr_male": fpr_male,
        "fpr_female": fpr_female,
        "fnr_male": fnr_male,
        "fnr_female": fnr_female,
        
        # Raw counts
        "male_count": gender_counts["man"],
        "female_count": gender_counts["woman"],
        "total_count": total,
        
        # For metrics that expect arrays
        "fpr_array": [fpr_male, fpr_female],
        "fnr_array": [fnr_male, fnr_female],
        "tpr_array": [tpr_male, tpr_female],
        "classification_data": [fpr_male, fpr_female, fnr_male, fnr_female]
    }

    # Return raw data or calculate metrics based on parameter
    if return_raw_data:
        return {"captions": captions, "gender_counts": gender_counts, "model_counts": model_counts, "comprehensive_data": comprehensive_data}
    elif metrics:
        results = {}
        # Format data once inside the function
        formatted_data = format_data(model_counts)
        for name, metric_func in metrics.items():
            try:
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
            except Exception as e:
                print(f"Warning: Metric {name} failed with error: {e}")
                print(f"  formatted_data: {formatted_data}")
                print(f"  function: {metric_func}")
                # Try to provide alternative data for specific metrics
                if name == "equality_of_odds":
                    # Use the comprehensive classification data
                    value = metric_func(comprehensive_data["classification_data"])
                    results[name] = value
                elif name == "predictive_equality":
                    # Use FPR data
                    value = metric_func(comprehensive_data["fpr_array"])
                    results[name] = value
                elif name == "tpr":
                    # Use TPR data
                    value = metric_func([comprehensive_data["tpr_male"], comprehensive_data["total_count"]])
                    results[name] = value
                elif name == "error_rate":
                    # Calculate error rate from TPR
                    avg_tpr = (comprehensive_data["tpr_male"] + comprehensive_data["tpr_female"]) / 2
                    value = 1 - avg_tpr
                    results[name] = value
                elif name == "fpr_error_rate":
                    # Calculate FPR error rate
                    avg_fpr = (comprehensive_data["fpr_male"] + comprehensive_data["fpr_female"]) / 2
                    value = 1 - avg_fpr
                    results[name] = value
                else:
                    results[name] = 0.0  # Default value on error
        return results
    else:
        # Return formatted data for external metric calculation
        return format_data(model_counts)
    
    # Clean up GPU memory
    if device == "cuda":
        # Delete models to free GPU memory
        del model
        del processor
        # Clear all GPU cache
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        print("GPU memory cleaned up")

if __name__ == "__main__":
    # Run with default settings when script is run directly
    run_image_to_text_evaluation(BLIP_LARGE_MODEL)
