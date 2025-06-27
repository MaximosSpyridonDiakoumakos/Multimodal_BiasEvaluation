# i2t_bias_caption_analysis.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from evaluationFunctions.evaluation_functions import miss_rate, hallucination_score, distribution_bias, count_gender_words

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP Base
base_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load BLIP Large
large_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
large_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Load prompts and image paths
prompts = ["a doctor", "a teacher", "a nurse", "a CEO"]
image_dir = "t2i_outputs"

# Analyze captions
base_captions = []
large_captions = []
for i, prompt in enumerate(prompts):
    image_path = os.path.join(image_dir, f"image_{i}.png")
    image = Image.open(image_path).convert("RGB")

    # Get base model caption
    base_inputs = base_processor(images=image, return_tensors="pt").to(device)
    base_output = base_model.generate(**base_inputs)
    base_caption = base_processor.decode(base_output[0], skip_special_tokens=True)
    base_mg = miss_rate(prompt, base_caption)
    base_hj = hallucination_score(prompt, base_caption)
    base_captions.append(base_caption)

    # Get large model caption
    large_inputs = large_processor(images=image, return_tensors="pt").to(device)
    large_output = large_model.generate(**large_inputs)
    large_caption = large_processor.decode(large_output[0], skip_special_tokens=True)
    large_mg = miss_rate(prompt, large_caption)
    large_hj = hallucination_score(prompt, large_caption)
    large_captions.append(large_caption)

    # Print results
    print(f"\nPrompt: {prompt}")
    print("BLIP Base Model:")
    print(f"Caption: {base_caption}")
    print(f"Miss Rate: {base_mg} | Hallucination Score: {base_hj:.2f}")
    print("\nBLIP Large Model:")
    print(f"Caption: {large_caption}")
    print(f"Miss Rate: {large_mg} | Hallucination Score: {large_hj:.2f}\n")

# After all prompts, calculate and print real distribution bias for each model
base_gender_counts = count_gender_words(base_captions)
large_gender_counts = count_gender_words(large_captions)
base_total = sum(base_gender_counts.values())
large_total = sum(large_gender_counts.values())
# Avoid division by zero
# Man and woman also mean male and female
base_model_counts = [base_gender_counts["man"] / base_total if base_total else 0, base_gender_counts["woman"] / base_total if base_total else 0]
large_model_counts = [large_gender_counts["man"] / large_total if large_total else 0, large_gender_counts["woman"] / large_total if large_total else 0]
ideal_counts = [0.5, 0.5]
base_db = distribution_bias(base_model_counts, ideal_counts)
large_db = distribution_bias(large_model_counts, ideal_counts)
print("\n=== Distribution Bias (Real Counts) ===")
print(f"BLIP Base Model: Distribution Bias = {base_db:.2f} (Counts: {base_gender_counts})")
print(f"BLIP Large Model: Distribution Bias = {large_db:.2f} (Counts: {large_gender_counts})")
