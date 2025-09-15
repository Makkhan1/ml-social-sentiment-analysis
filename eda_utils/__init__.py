"""
Exploratory Data Analysis utilities for sentiment analysis

This package provides visualization and analysis tools for sentiment analysis
workflows, including data exploration, preprocessing insights, and model
evaluation visualizations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import all utility functions
from .eda_utils import (
    setup_plotting,
    plot_initial_data_exploration,
    plot_preprocessing_insights,
    plot_data_splits,
    plot_comment_length_distribution
)

# Define what gets imported with "from package import *"
__all__ = [
    'setup_plotting',
    'plot_initial_data_exploration',
    'plot_preprocessing_insights',
    'plot_data_splits',
    'plot_comment_length_distribution',
    # Also expose the main libraries for convenience
    'plt',
    'sns',
    'pd',
    'np'
]

# Package-level configuration
def configure_plots(style='default', palette='husl', figsize=(10, 6), fontsize=12):
    """
    Configure global plotting settings for the package
    
    Args:
        style (str): Matplotlib style to use
        palette (str): Seaborn color palette
        figsize (tuple): Default figure size
        fontsize (int): Default font size
    """
    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize

# Initialize with default settings
configure_plots()

# Package information
def get_info():
    """Return package information"""
    return {
        'name': 'Sentiment Analysis EDA Utils',
        'version': __version__,
        'description': __doc__.strip(),
        'functions': __all__[:-4]  # Exclude the library imports
    }

print(f"Sentiment Analysis EDA Utils v{__version__} loaded successfully")
print("Available functions:", ", ".join(__all__[:-4]))

