"""
Data Processing Package for ML Sentiment Analysis
Contains utilities for data preprocessing, EDA, and visualization
"""

__version__ = "1.0.0"
__author__ = "Sachin Kumar"
__email__ = "2021ugpi047@nitjsr.ac.in"

# Import main classes and functions
try:
    from .processor import (
        DataPreprocessor,
        load_data_from_url,
        extract_comments,
        create_ensemble_sentiment
    )
    
    from .eda_util import (
        setup_plotting,
        plot_initial_data_exploration,
        plot_preprocessing_insights,
        plot_data_splits,
        plot_comment_length_distribution
    )
    
    print("✅ Data processing package loaded successfully!")
    
except ImportError as e:
    print(f"⚠️  Warning: Some modules could not be imported: {e}")
    
    # Fallback imports - try to import what we can
    try:
        from .processor import DataPreprocessor
        print("✅ DataPreprocessor loaded")
    except ImportError:
        print("❌ Could not load DataPreprocessor")
    
    try:
        from .eda_util import setup_plotting
        print("✅ EDA utilities loaded")
    except ImportError:
        print("❌ Could not load EDA utilities")

# Package metadata
__all__ = [
    # Main classes
    'DataPreprocessor',
    
    # Data loading and processing functions
    'load_data_from_url',
    'extract_comments', 
    'create_ensemble_sentiment',
    
    # Plotting and visualization functions
    'setup_plotting',
    'plot_initial_data_exploration',
    'plot_preprocessing_insights', 
    'plot_data_splits',
    'plot_comment_length_distribution'
]

# Package information
PACKAGE_INFO = {
    'name': 'ml-sentiment-data-processing',
    'version': __version__,
    'description': 'Data processing utilities for ML sentiment analysis',
    'author': __author__,
    'email': __email__,
    'components': {
        'DataPreprocessor': 'Main class for text preprocessing and sentiment analysis',
        'EDA utilities': 'Functions for exploratory data analysis and visualization',
        'Data loaders': 'Functions for loading and processing data from various sources'
    }
}

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def test_imports():
    """Test if all components can be imported successfully"""
    results = {}
    
    # Test processor imports
    try:
        from .processor import DataPreprocessor
        results['DataPreprocessor'] = True
    except ImportError:
        results['DataPreprocessor'] = False
    
    try:
        from .processor import load_data_from_url, extract_comments, create_ensemble_sentiment
        results['processor_functions'] = True
    except ImportError:
        results['processor_functions'] = False
    
    # Test EDA imports  
    try:
        from .eda_util import setup_plotting, plot_initial_data_exploration
        results['eda_functions'] = True
    except ImportError:
        results['eda_functions'] = False
    
    return results

# Quick usage example
USAGE_EXAMPLE = """
# Quick Usage Example:

from data import DataPreprocessor, load_data_from_url, setup_plotting

# 1. Load data
df = load_data_from_url('your_data_url_here')

# 2. Initialize preprocessor  
preprocessor = DataPreprocessor()

# 3. Clean text
cleaned_text = preprocessor.clean_text("Your text here!")

# 4. Setup plotting
setup_plotting()

# 5. Create visualizations
plot_initial_data_exploration(df, comment_columns)
"""

def show_usage():
    """Show package usage example"""
    print(USAGE_EXAMPLE)

# Auto-run package info on import (optional)
if __name__ != "__main__":
    pass  # Uncomment next line if you want info on every import
    # print(f"📦 Loaded {PACKAGE_INFO['name']} v{__version__}")
