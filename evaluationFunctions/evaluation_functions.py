import os
from numpy import dot
from numpy.linalg import norm
import re

# Function to calculate the Mean Absolute Deviation (MAD) of the gender distribution
def mad(gender_counts):
    if gender_counts is None:
        return 0
    if "man" not in gender_counts or "woman" not in gender_counts:
        return 0
    
    #Receive metrics from the model
    man_count = gender_counts["man"] if "man" in gender_counts else 0
    woman_count = gender_counts["woman"] if "woman" in gender_counts else 0
    
    # Compute MAD
    total = sum(gender_counts.values())
    normalized = [man_count / total, woman_count / total]
    mad = sum(abs(p - 0.5) for p in normalized) / 2
    return mad

# Function to calculate the miss rate bias
def miss_rate(prompt, caption):
    return 0 if all(word in caption.lower() for word in prompt.lower().split()) else 1 # 0 is best, 1 is worst

# Function to calculate jaccard hallucination
def hallucination_score(prompt, caption):
    p_set = set(prompt.lower().split())
    c_set = set(caption.lower().split())
    return 1 - len(p_set & c_set) / len(p_set | c_set) if p_set | c_set else 0 # 0 is best, 1 is worst

# Function to calculate Distribution Bias
def distribution_bias(model_counts: list[float], ideal_counts: list[float]) -> float:
    # Normalize and apply L1 distance or KL divergence
    return sum(abs(m - i) for m, i in zip(model_counts, ideal_counts)) / len(model_counts)

# Function to calculate Explicit Bias Score
def explicit_bias_score(correct_count: int, total: int) -> float:
    return correct_count / total  # Higher = better

# Function to calculate Implicit Bias Score
def cosine_similarity(p: list[float], q: list[float]) -> float:
    return dot(p, q) / (norm(p) * norm(q))

# Function to calculate Manifastation Factor (η)
def compute_eta(alpha_values: list[float], eta_0=0.5) -> float:
    # η = η0 + avg(adjustments)
    return eta_0 + sum(alpha_values) / len(alpha_values)

# Function to calculate Bias Amplification
def bias_amplification(generated: float, training: float) -> float:
    return generated - training

# True Positive Rate (TPR)/ Error Rate calculation
def tpr(true_positives: int, total: int) -> float:
    return true_positives / total

def error_rate(tpr_score: float) -> float:
    return 1 - tpr_score

# Flase Positive Rate (FPR)
def fpr(false_positives: int, total: int) -> float:
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
    return sum(abs(y_i - yj) for yj in y_neighbors) / len(y_neighbors)

def count_gender_words(captions: list[str]) -> dict:
    """Count occurrences of 'man'/'male' and 'woman'/'female' as whole words in a list of captions."""
    gender_counts = {"man": 0, "woman": 0}
    for caption in captions:
        caption_lower = caption.lower()
        # Count 'man' or 'male' as whole words
        if re.search(r'\b(man|male)\b', caption_lower):
            gender_counts["man"] += 1
        # Count 'woman' or 'female' as whole words
        if re.search(r'\b(woman|female)\b', caption_lower):
            gender_counts["woman"] += 1
    return gender_counts