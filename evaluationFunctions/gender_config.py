# gender_config.py
# Configuration for gender terms used in evaluation functions

# Gender terms dictionary for consistent usage across evaluation functions
GENDER_TERMS = {
    "male": ["man", "male"],
    "female": ["woman", "female"]
}

def get_gender_terms(gender_type: str) -> list:
    """
    Get gender terms for a specific gender type.
    
    Args:
        gender_type: "male" or "female"
        
    Returns:
        List of terms for that gender type
    """
    return GENDER_TERMS.get(gender_type, []) 