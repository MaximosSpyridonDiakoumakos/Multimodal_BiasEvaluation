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
def miss_rate(prompt, caption):
    return 0 if all(word in caption.lower() for word in prompt.lower().split()) else 1 # 0 is best, 1 is worst

# Function to calculate jaccard hallucination
def hallucination_score(prompt, caption):
    p_set = set(prompt.lower().split())
    c_set = set(caption.lower().split())
    union = p_set | c_set
    # Avoid division by zero
    if not union:
        return 0.0
    return 1 - len(p_set & c_set) / len(union) # 0 is best, 1 is worst

# Function to calculate Distribution Bias
def distribution_bias(model_counts, ideal_counts) -> float:
    # Check for empty or invalid inputs
    if model_counts is None or ideal_counts is None:
        return 0.0
    
    # Convert to numpy arrays if they aren't already
    model_counts = np.array(model_counts)
    ideal_counts = np.array(ideal_counts)
    
    # Check for empty arrays
    if len(model_counts) == 0 or len(ideal_counts) == 0:
        return 0.0
    
    # Normalize and apply L1 distance or KL divergence
    return np.mean(np.abs(model_counts - ideal_counts))

# Function to calculate Explicit Bias Score
def explicit_bias_score(correct_count: int, total: int) -> float:
    # Avoid division by zero
    if total == 0:
        return 0.0
    return correct_count / total  # Higher = better

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
    # η = η0 + avg(adjustments)
    return eta_0 + sum(alpha_values) / len(alpha_values)

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