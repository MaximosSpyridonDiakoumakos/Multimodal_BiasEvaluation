"""
Multimodal Bias Evaluation Package

A comprehensive toolkit for evaluating bias in multimodal AI systems,
specifically focused on text-to-image and image-to-text models.

Author: Maximos Spyridon Diakoumakos
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Maximos Spyridon Diakoumakos"
__email__ = "it2021027@hua.gr"

# Import main functions for easy access
from .main import main
from .text_to_image import run_text_to_image_evaluation
from .image_to_text import run_image_to_text_evaluation

# Import evaluation functions
from .evaluationFunctions.evaluation_functions import (
    mad, distribution_bias, miss_rate, hallucination_score,
    format_data, count_gender_words
)

# Import visualization functions
from .evaluationFunctions.visualization import (
    plot_gender_distribution, plot_metric_comparison,
    plot_model_comparison, create_summary_report
)

# Import configuration
from .prompts_config import (
    PROMPTS, GENDER_LABELS, IMAGE_DIR,
    STABLE_DIFFUSION_MODEL_v1_5, STABLE_DIFFUSION_XL_MODEL, STABLE_DIFFUSION_V1_4_MODEL,
    BLIP_BASE_MODEL, BLIP_LARGE_MODEL, CLIP_MODEL
)

__all__ = [
    # Main functions
    'main',
    'run_text_to_image_evaluation',
    'run_image_to_text_evaluation',
    
    # Evaluation functions
    'mad',
    'distribution_bias', 
    'miss_rate',
    'hallucination_score',
    'format_data',
    'count_gender_words',
    
    # Visualization functions
    'plot_gender_distribution',
    'plot_metric_comparison',
    'plot_model_comparison',
    'create_summary_report',
    
    # Configuration
    'PROMPTS',
    'GENDER_LABELS',
    'IMAGE_DIR',
    'STABLE_DIFFUSION_MODEL_v1_5',
    'STABLE_DIFFUSION_XL_MODEL',
    'STABLE_DIFFUSION_V1_4_MODEL',
    'BLIP_BASE_MODEL',
    'BLIP_LARGE_MODEL',
    'CLIP_MODEL'
] 