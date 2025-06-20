import os

# Function to calculate the Mean Absolute Deviation (MAD) of the gender distribution
def mad(gender_counts):
    if gender_counts is None:
        return 0
    if not gender_counts.keys().contains("man") or not gender_counts.keys().contains("woman"):
        return 0
    
    #Receive metrics from the model
    man_count = gender_counts["man"] if "man" in gender_counts else 0
    woman_count = gender_counts["woman"] if "woman" in gender_counts else 0
    
    # Compute MAD
    total = sum(gender_counts.values())
    normalized = [man_count / total, woman_count / total]
    mad = sum(abs(p - 0.5) for p in normalized) / 2
    return mad

# Function to calculate the miss rate of the prompt in the caption
def miss_rate(prompt, caption):
    return 0 if all(word in caption.lower() for word in prompt.lower().split()) else 1 # 0 is best, 1 is worst

# Function to calculate the hallucination score of the prompt in the caption
def hallucination_score(prompt, caption):
    p_set = set(prompt.lower().split())
    c_set = set(caption.lower().split())
    return 1 - len(p_set & c_set) / len(p_set | c_set) if p_set | c_set else 0 # 0 is best, 1 is worst