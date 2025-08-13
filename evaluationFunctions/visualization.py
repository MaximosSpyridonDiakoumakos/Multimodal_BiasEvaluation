# visualization.py
# Visualization functions for evaluation results

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any

def setup_colab_display():
    """Set up matplotlib for optimal display in Google Colab."""
    try:
        # Check if we're in Colab
        import google.colab
        # Set matplotlib backend for Colab
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        print("Google Colab detected - matplotlib configured for optimal display")
    except ImportError:
        print("Not in Google Colab - using standard matplotlib settings")

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # Colab-specific optimizations
    try:
        import google.colab
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
    except ImportError:
        pass

def plot_gender_distribution(gender_counts: Dict[str, int], title: str = "Gender Distribution"):
    """
    Plot gender distribution as a bar chart.
    
    Args:
        gender_counts: Dictionary with gender counts
        title: Title for the plot
    """
    setup_plot_style()
    
    genders = list(gender_counts.keys())
    counts = list(gender_counts.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(genders, counts, color=['#2E86AB', '#A23B72'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Gender', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Adjust y-axis to accommodate value labels
    y_max = max(counts)
    plt.ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    plt.tight_layout()
    plt.show()

def plot_metric_comparison(metrics_data: Dict[str, float], title: str = "Metric Comparison"):
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_data: Dictionary with metric names and values
        title: Title for the plot
    """
    setup_plot_style()
    
    # Helper function to safely convert numpy values to Python scalars
    def safe_convert(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))  # Take mean for multi-element arrays
        elif isinstance(value, np.generic):
            return float(value.item())
        else:
            return float(value)
    
    metrics = list(metrics_data.keys())
    values = [safe_convert(value) for value in metrics_data.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color='skyblue', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust y-axis to accommodate value labels
    y_max = max(values)
    plt.ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_results: Dict[str, Dict[str, float]], metric_name: str):
    """
    Plot comparison of different models for a specific metric.
    
    Args:
        model_results: Dictionary with model names and their metric results
        metric_name: Name of the metric to compare
    """
    setup_plot_style()
    
    # Helper function to safely convert numpy values to Python scalars
    def safe_convert(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))  # Take mean for multi-element arrays
        elif isinstance(value, np.generic):
            return float(value.item())
        else:
            return float(value)
    
    models = list(model_results.keys())
    values = [safe_convert(model_results[model].get(metric_name, 0)) for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='lightcoral', alpha=0.7)
    plt.title(f'{metric_name} Comparison Across Models', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel('Models', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust y-axis to accommodate value labels
    y_max = max(values)
    plt.ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_bias_heatmap(bias_data: Dict[str, Dict[str, float]], title: str = "Bias Heatmap"):
    """
    Create a heatmap showing bias scores across different models and metrics.
    
    Args:
        bias_data: Dictionary with model names and their bias metrics
        title: Title for the plot
    """
    setup_plot_style()
    
    # Helper function to safely convert numpy values to Python scalars
    def safe_convert(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))  # Take mean for multi-element arrays
        elif isinstance(value, np.generic):
            return float(value.item())
        else:
            return float(value)
    
    # Prepare data for heatmap
    models = list(bias_data.keys())
    metrics = set()
    for model_data in bias_data.values():
        metrics.update(model_data.keys())
    metrics = list(metrics)
    
    # Create matrix for heatmap
    heatmap_data = []
    for model in models:
        row = [safe_convert(bias_data[model].get(metric, 0)) for metric in metrics]
        heatmap_data.append(row)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=metrics, 
                yticklabels=models,
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_summary_report(results: Dict[str, Any], save_path: str = None):
    """
    Create a comprehensive summary report with multiple visualizations.
    
    Args:
        results: Dictionary containing all evaluation results
        save_path: Optional path to save the report
    """
    setup_plot_style()
    
    # Helper function to safely convert numpy values to Python scalars
    def safe_convert(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))  # Take mean for multi-element arrays
        elif isinstance(value, np.generic):
            return float(value.item())
        else:
            return float(value)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multimodal Bias Evaluation Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Text-to-Image MAD scores
    if 'text_to_image' in results:
        t2i_models = list(results['text_to_image'].keys())
        t2i_mad_scores = []
        for model in t2i_models:
            mad_value = results['text_to_image'][model].get('mad', 0)
            t2i_mad_scores.append(safe_convert(mad_value))
        
        if t2i_mad_scores:  # Only plot if we have valid scores
            axes[0, 0].bar(t2i_models, t2i_mad_scores, color='lightblue')
            axes[0, 0].set_title('Text-to-Image MAD Scores')
            axes[0, 0].set_ylabel('MAD Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
                    # Add value labels
        for i, score in enumerate(t2i_mad_scores):
            axes[0, 0].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        # Adjust y-axis to accommodate value labels
        y_max = max(t2i_mad_scores)
        axes[0, 0].set_ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    # Plot 2: Image-to-Text Metrics
    if 'image_to_text' in results:
        i2t_results = results['image_to_text']
        if i2t_results:
            # Get the first model's results to extract metric names
            first_model = list(i2t_results.keys())[0]
            i2t_metrics = list(i2t_results[first_model].keys())
            
            # Calculate average scores across models for each metric
            metric_scores = {}
            for metric in i2t_metrics:
                scores = []
                for model in i2t_results.keys():
                    metric_value = i2t_results[model].get(metric, 0)
                    scores.append(safe_convert(metric_value))
                metric_scores[metric] = sum(scores) / len(scores)
            
            # Plot the metrics
            metrics = list(metric_scores.keys())
            scores = list(metric_scores.values())
            
            if scores:  # Only plot if we have valid scores
                axes[0, 1].bar(metrics, scores, color='lightgreen')
                axes[0, 1].set_title('Image-to-Text Metrics')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for i, score in enumerate(scores):
                    axes[0, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
                
                # Adjust y-axis to accommodate value labels
                y_max = max(scores)
                axes[0, 1].set_ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    # Plot 3: Overall Model Comparison (MAD scores across all models)
    all_models = []
    all_mad_scores = []
    
    # Collect MAD scores from text-to-image models
    if 'text_to_image' in results:
        for model_name, model_results in results['text_to_image'].items():
            mad_value = model_results.get('mad', 0)
            all_models.append(f"T2I: {model_name}")
            all_mad_scores.append(safe_convert(mad_value))
    
    # Collect MAD scores from image-to-text models
    if 'image_to_text' in results:
        for model_name, model_results in results['image_to_text'].items():
            mad_value = model_results.get('mad', 0)
            all_models.append(f"I2T: {model_name}")
            all_mad_scores.append(safe_convert(mad_value))
    
    if all_mad_scores:  # Only plot if we have valid scores
        bars = axes[1, 0].bar(all_models, all_mad_scores, 
                              color=['lightblue' if 'T2I:' in model else 'lightgreen' for model in all_models])
        axes[1, 0].set_title('Overall Model Comparison (MAD Scores)')
        axes[1, 0].set_ylabel('MAD Score')
        axes[1, 0].tick_params(axis='x', rotation=45, ha='right')
        
        # Add value labels
        for i, score in enumerate(all_mad_scores):
            axes[1, 0].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        # Adjust y-axis to accommodate value labels
        y_max = max(all_mad_scores)
        axes[1, 0].set_ylim(0, y_max * 1.15)  # Add 15% padding above the highest bar
    
    # Plot 4: Summary Statistics (Bias Score Distribution)
    bias_scores = []
    bias_labels = []
    
    # Collect bias-related metrics from all models
    if 'text_to_image' in results:
        for model_name, model_results in results['text_to_image'].items():
            # Get bias-related metrics
            explicit_bias = safe_convert(model_results.get('explicit_bias_score', 0))
            implicit_bias = safe_convert(model_results.get('implicit_bias_score', 0))
            distribution_bias = safe_convert(model_results.get('distribution_bias', 0))
            
            bias_scores.extend([explicit_bias, implicit_bias, distribution_bias])
            bias_labels.extend([f'{model_name}_explicit', f'{model_name}_implicit', f'{model_name}_dist'])
    
    if 'image_to_text' in results:
        for model_name, model_results in results['image_to_text'].items():
            # Get bias-related metrics
            distribution_bias = safe_convert(model_results.get('distribution_bias', 0))
            demographic_parity = safe_convert(model_results.get('demographic_parity', 0))
            
            bias_scores.extend([distribution_bias, demographic_parity])
            bias_labels.extend([f'{model_name}_dist', f'{model_name}_demo'])
    
    if bias_scores:  # Only plot if we have valid scores
        # Create a horizontal bar chart for better readability
        y_pos = range(len(bias_scores))
        bars = axes[1, 1].barh(y_pos, bias_scores, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Bias Score Distribution Across Models')
        axes[1, 1].set_xlabel('Bias Score')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(bias_labels, fontsize=8)
        
        # Add value labels
        for i, score in enumerate(bias_scores):
            axes[1, 1].text(score + 0.01, i, f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    # Adjust subplot spacing to prevent overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_gender_proportions(gender_counts: Dict[str, int], title: str = "Gender Proportions"):
    """
    Plot gender proportions as a pie chart.
    
    Args:
        gender_counts: Dictionary with gender counts
        title: Title for the plot
    """
    setup_plot_style()
    
    labels = list(gender_counts.keys())
    sizes = list(gender_counts.values())
    colors = ['#2E86AB', '#A23B72']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.show() 