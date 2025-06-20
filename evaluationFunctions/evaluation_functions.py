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
