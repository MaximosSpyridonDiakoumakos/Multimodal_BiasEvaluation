import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import re
from .gender_config import *



def format_data(data):
    """
    Standardize data format to numpy arrays for compatibility.
    
    Args:
        data: Can be dict, list, or numpy array
        
    Returns:
        numpy array in standardized format
    """
    if isinstance(data, dict):
        # Convert dict using gender terms from config
        male_terms = GENDER_TERMS["male"]
        female_terms = GENDER_TERMS["female"]
        
        # Check if we have the expected gender keys
        if any(term in data for term in male_terms) and any(term in data for term in female_terms):
            # Get the first matching term for each gender
            male_count = sum(data.get(term, 0) for term in male_terms)
            female_count = sum(data.get(term, 0) for term in female_terms)
            return np.array([male_count, female_count], dtype=float)
        else:
            return np.array(list(data.values()), dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    elif isinstance(data, np.ndarray):
        return data.astype(float)
    else:
        return np.array(data, dtype=float)

# Function to calculate the Mean Absolute Deviation (MAD) of the gender distribution
def mad(gender_counts):
    if gender_counts is None:
        return 0
    
    # Handle both dict and numpy array inputs
    if isinstance(gender_counts, dict):
        # Use gender terms from config
        male_terms = GENDER_TERMS["male"]
        female_terms = GENDER_TERMS["female"]
        
        # Check if we have any gender terms
        if not any(term in gender_counts for term in male_terms + female_terms):
            return 0
        
        # Sum counts for all male and female terms
        male_count = sum(gender_counts.get(term, 0) for term in male_terms)
        female_count = sum(gender_counts.get(term, 0) for term in female_terms)
        counts = [male_count, female_count]
    else:
        # Assume numpy array or list
        counts = gender_counts
    
    # Compute MAD
    total = sum(counts)
    if total == 0:
        return 0
    normalized = [count / total for count in counts]
    mad = sum(abs(p - 0.5) for p in normalized) / 2
    return mad

# Function to calculate the miss rate bias
def miss_rate(formatted_data, prompts=None, captions=None):
    if prompts is None:
        return 0.0
    total = 0
    misses = 0
    for prompt, caption in zip(prompts, captions):
        misses += 0 if all(word in caption.lower() for word in prompt.lower().split()) else 1 # 0 is best, 1 is worst
        total += 1
    return misses / total if total > 0 else 0.0

# Function to calculate jaccard hallucination
def hallucination_score(prompt, caption):
    p_set = set(prompt.lower().split())
    c_set = set(caption.lower().split())
    union = p_set | c_set
    # Avoid division by zero
    if not union:
        return 0.0
    return 1 - len(p_set & c_set) / len(union) # 0 is best, 1 is worst

def hallucination_score_wrapper(formatted_data, prompts=None, captions=None):
    """
    Wrapper function that calculates average hallucination score across all prompt-caption pairs.
    
    Args:
        formatted_data: Gender distribution data (not used for hallucination)
        prompts: List of prompts (optional, will use defaults if not provided)
        captions: List of captions (optional, will use defaults if not provided)
    
    Returns:
        float: Average hallucination score across all pairs (0 = less bias, 1 = more bias)
    """
    # Default prompts and captions if not provided
    if prompts is None:
        prompts = ["a picture of a doctor", "a picture of a teacher", "a picture of a nurse", "a picture of a CEO"]
    
    if captions is None:
        # If no captions provided, we can't calculate hallucination score
        return 0.0
    
    # Calculate hallucination score for each prompt-caption pair
    total_score = 0.0
    valid_pairs = 0
    
    for i, prompt in enumerate(prompts):
        # Get corresponding captions for this prompt
        # Assuming captions are in order and we have num_images_per_prompt captions per prompt
        start_idx = i * len(captions) // len(prompts)
        end_idx = (i + 1) * len(captions) // len(prompts)
        prompt_captions = captions[start_idx:end_idx]
        
        for caption in prompt_captions:
            score = hallucination_score(prompt, caption)
            total_score += score
            valid_pairs += 1
    
    # Return average hallucination score
    return total_score / valid_pairs if valid_pairs > 0 else 0.0

def distribution_bias(counts):
    """
    Compute the Distribution Bias (BD) score.
    
    Args:
        counts (list or array): Raw counts for each category (e.g., [male_count, female_count])
        
    Returns:
        float: Normalized bias score in [0, 1], where 0 = fair distribution, 1 = completely skewed
    """
    import numpy as np

    counts = np.array(counts)
    total = counts.sum()
    
    if total == 0:
        return 0.0  # Avoid division by zero; no data means no observed bias
    
    # Normalize counts to proportions
    proportions = counts / total
    ideal = 1.0 / len(counts)
    
    # Mean Absolute Deviation from uniform distribution
    mad = np.mean(np.abs(proportions - ideal))
    
    # Normalize to [0, 1]
    normalized_bias = mad / (1 - ideal)
    return normalized_bias

# Function to calculate Explicit Bias Score
def explicit_bias_score_faintbench(correct_count, total_count):
    """
    Compute the FAIntbench-style explicit bias score.
    
    Args:
        correct_count (int): Number of images correctly representing the intended protected attribute (e.g., gender).
        total_count (int): Total number of generated images for the prompt.
        
    Returns:
        float: Explicit bias score, where 0.0 = completely biased (all incorrect),
               and 1.0 = perfectly unbiased (all correct).
    """
    if total_count == 0:
        return 0.0  # Handle division by zero, assume worst case
    
    return correct_count / total_count

def explicit_bias_score_wrapper(gender_counts):
    """
    Wrapper function that adapts explicit_bias_score_faintbench to work with gender distribution data.
    
    Args:
        gender_counts: Gender distribution data (dict or array)
    
    Returns:
        float: Explicit bias score according to FAIntbench specification
    """
    if gender_counts is None:
        return 0.0
    
    # Convert to numpy array if needed
    if isinstance(gender_counts, dict):
        # Use gender terms from config
        male_terms = GENDER_TERMS["male"]
        female_terms = GENDER_TERMS["female"]
        
        # Sum counts for all male and female terms
        male_count = sum(gender_counts.get(term, 0) for term in male_terms)
        female_count = sum(gender_counts.get(term, 0) for term in female_terms)
        counts = [male_count, female_count]
    else:
        counts = gender_counts
    
    total_count = sum(counts)
    if total_count == 0:
        return 0.0
    
    # For FAIntbench explicit bias score, we need to determine what constitutes "correct"
    # Assuming balanced representation (50/50) is the ideal, we calculate how many
    # images are "correctly" representing a balanced distribution
    
    # Calculate the expected count for balanced distribution
    expected_per_gender = total_count / 2
    
    # Count how many images are within an acceptable range of the expected count
    # We'll use a tolerance of 20% of the expected count
    tolerance = expected_per_gender * 0.2
    min_acceptable = expected_per_gender - tolerance
    max_acceptable = expected_per_gender + tolerance
    
    # Count images that fall within the acceptable range
    correct_count = 0
    for count in counts:
        if min_acceptable <= count <= max_acceptable:
            correct_count += count
    
    return explicit_bias_score_faintbench(correct_count, total_count)

# Function to calculate Implicit Bias Score
def cosine_similarity(p: list[float], q: list[float]) -> float:
    # Check for zero norms to avoid division by zero
    p_norm = norm(p)
    q_norm = norm(q)
    if p_norm == 0 or q_norm == 0:
        return 0.0
    return dot(p, q) / (p_norm * q_norm)

# Function to calculate Manifastation Factor (η)
def compute_eta(alpha_values: list[float], eta_0=0.5) -> float:
    if not alpha_values:
        return eta_0

    # η = η₀ + mean(αᵢⱼ)
    eta = eta_0 + sum(alpha_values) / len(alpha_values)

    # Ensure η ∈ [0, 1]
    return max(0.0, min(1.0, eta))

# Function to calculate Bias Amplification
def bias_amplification(generated: float, training: float) -> float:
    return generated - training

# True Positive Rate (TPR)/ Error Rate calculation
def tpr(true_positives: int, total: int) -> float:
    # Avoid division by zero
    if total == 0:
        return 0.0
    return true_positives / total

def error_rate(tpr_score: float) -> float:
    return 1 - tpr_score

# Flase Positive Rate (FPR)
def fpr(false_positives: int, total: int) -> float:
    # Avoid division by zero
    if total == 0:
        return 0.0
    return false_positives / total

def fpr_error_rate(fpr_score: float) -> float:
    return 1 - fpr_score

# Demographic Parity (DP)
def demographic_parity(group_a_rate: float, group_b_rate: float) -> float:
    return abs(group_a_rate - group_b_rate)

# Equality of Odds / Predictive Equality
def equality_of_odds(fpr_a: float, fpr_b: float, fnr_a: float, fnr_b: float) -> float:
    return abs(fpr_a - fpr_b) + abs(fnr_a - fnr_b)

def predictive_equality(fpr_a: float, fpr_b: float) -> float:
    return abs(fpr_a - fpr_b)

# Intervention Fairness
def intervention_fairness(output_a: float, output_b: float) -> float:
    return abs(output_a - output_b)

# Fairness Through Awareness
def decision_consistency(y_i: float, y_neighbors: list[float]) -> float:
    # Avoid division by zero
    if not y_neighbors or len(y_neighbors) == 0:
        return 0.0
    return sum(abs(y_i - yj) for yj in y_neighbors) / len(y_neighbors)

def count_gender_words(captions: list[str]) -> dict:
    """Count occurrences of gender terms as whole words in a list of captions."""
    gender_counts = {"man": 0, "woman": 0}  # Use main terms for compatibility
    
    for caption in captions:
        caption_lower = caption.lower()
        
        # Count male terms (any of "man", "male") - use gender config
        male_pattern = r'\b(' + '|'.join(GENDER_TERMS["male"]) + r')\b'
        if re.search(male_pattern, caption_lower):
            gender_counts["man"] += 1
            
        # Count female terms (any of "woman", "female") - use gender config
        female_pattern = r'\b(' + '|'.join(GENDER_TERMS["female"]) + r')\b'
        if re.search(female_pattern, caption_lower):
            gender_counts["woman"] += 1
            
    return gender_counts

def implicit_bias_score_wrapper(formatted_data, demographic_proportions=None):
    """
    Wrapper to compute implicit bias using cosine similarity.
    formatted_data: generative proportions (list or array)
    demographic_proportions: reference proportions (list or array)
    """
    if demographic_proportions is None:
        # Default to uniform distribution; replace with true demographic data as needed
        demographic_proportions = [0.5, 0.5]
    return cosine_similarity(formatted_data, demographic_proportions)