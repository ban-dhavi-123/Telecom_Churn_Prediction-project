"""
Configuration File for Telecom Churn Prediction Project
========================================================
This file contains all configurable parameters for the project.
Modify these settings to customize the behavior of the pipeline.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Models directory
MODELS_DIR = BASE_DIR / "models"

# Plots directory
PLOTS_DIR = BASE_DIR / "plots"

# Scripts directory
SCRIPTS_DIR = BASE_DIR / "scripts"

# Notebooks directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset path (update this with your actual dataset path)
DATASET_PATH = DATA_DIR / "telecom_churn.csv"  # or .xlsx

# Alternative: Use absolute path if dataset is elsewhere
# DATASET_PATH = r"C:\Users\lssan\Downloads\P585 Churn.xlsx"

# Target column name
TARGET_COLUMN = "Churn"

# Columns to exclude from modeling (e.g., customer ID)
EXCLUDE_COLUMNS = []  # Example: ['CustomerID', 'Name']

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Train-test split ratio
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# Stratify split by target variable
STRATIFY = True

# Missing value strategy
MISSING_VALUE_STRATEGY = {
    'numerical': 'median',  # Options: 'mean', 'median', 'mode', 'drop'
    'categorical': 'mode'   # Options: 'mode', 'drop', 'constant'
}

# Encoding strategy
ENCODING_STRATEGY = {
    'binary': 'label',      # Options: 'label', 'onehot'
    'multiclass': 'onehot'  # Options: 'onehot', 'label'
}

# Feature scaling
SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust', 'none'

# Handle class imbalance
HANDLE_IMBALANCE = False  # Set to True to use SMOTE
IMBALANCE_METHOD = 'smote'  # Options: 'smote', 'adasyn', 'random_oversample'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'solver': 'liblinear',
    'random_state': RANDOM_STATE,
    'C': 1.0
}

# SVM parameters
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_STATE
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Cross-validation folds
CV_FOLDS = 5

# Enable hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING = False

# Hyperparameter tuning method
TUNING_METHOD = 'grid'  # Options: 'grid', 'random', 'bayesian'

# Number of iterations for random search
RANDOM_SEARCH_ITERATIONS = 20

# Scoring metric for model selection
SCORING_METRIC = 'roc_auc'  # Options: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metrics to calculate
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5

# Generate confusion matrix
GENERATE_CONFUSION_MATRIX = True

# Generate ROC curve
GENERATE_ROC_CURVE = True

# Generate feature importance plot
GENERATE_FEATURE_IMPORTANCE = True

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot style
PLOT_STYLE = 'whitegrid'  # Options: 'whitegrid', 'darkgrid', 'white', 'dark', 'ticks'

# Figure size
FIGURE_SIZE = (12, 6)

# DPI for saved plots
PLOT_DPI = 300

# Plot format
PLOT_FORMAT = 'png'  # Options: 'png', 'jpg', 'pdf', 'svg'

# Color palette
COLOR_PALETTE = 'Set2'  # Options: 'Set1', 'Set2', 'Set3', 'Paired', etc.

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

# App title
APP_TITLE = "Telecom Customer Churn Prediction"

# App icon
APP_ICON = "📱"

# Page layout
PAGE_LAYOUT = "wide"  # Options: 'centered', 'wide'

# Theme
THEME = {
    'primaryColor': '#4CAF50',
    'backgroundColor': '#FFFFFF',
    'secondaryBackgroundColor': '#F0F2F6',
    'textColor': '#262730',
    'font': 'sans serif'
}

# Default model for predictions
DEFAULT_MODEL = 'best_model.pkl'

# Maximum file upload size (MB)
MAX_UPLOAD_SIZE = 200

# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================

# Docker image name
DOCKER_IMAGE_NAME = 'telecom-churn-prediction'

# Docker container port
DOCKER_PORT = 8501

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Enable logging
ENABLE_LOGGING = True

# Log level
LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# Log file path
LOG_FILE = BASE_DIR / 'logs' / 'churn_prediction.log'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

# Use parallel processing
USE_PARALLEL = True

# Number of parallel jobs (-1 for all cores)
N_JOBS = -1

# Batch size for predictions
BATCH_SIZE = 1000

# Cache preprocessed data
CACHE_PREPROCESSED_DATA = True

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration settings.
    """
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR, BASE_DIR / 'logs']:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"⚠ Warning: Dataset not found at {DATASET_PATH}")
        print("Please place your dataset in the data/ folder or update DATASET_PATH in config.py")
    
    # Validate test size
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1")
    
    # Validate CV folds
    if CV_FOLDS < 2:
        raise ValueError("CV_FOLDS must be at least 2")
    
    print("✓ Configuration validated successfully!")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(model_name):
    """
    Get full path for a model file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model file
    
    Returns:
    --------
    Path : Full path to model file
    """
    return MODELS_DIR / model_name


def get_plot_path(plot_name):
    """
    Get full path for a plot file.
    
    Parameters:
    -----------
    plot_name : str
        Name of the plot file
    
    Returns:
    --------
    Path : Full path to plot file
    """
    return PLOTS_DIR / plot_name


def get_data_path(data_name):
    """
    Get full path for a data file.
    
    Parameters:
    -----------
    data_name : str
        Name of the data file
    
    Returns:
    --------
    Path : Full path to data file
    """
    return DATA_DIR / data_name


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# Dictionary of all configuration parameters
CONFIG = {
    'paths': {
        'base_dir': BASE_DIR,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'plots_dir': PLOTS_DIR,
        'dataset_path': DATASET_PATH
    },
    'data': {
        'target_column': TARGET_COLUMN,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'stratify': STRATIFY
    },
    'preprocessing': {
        'missing_value_strategy': MISSING_VALUE_STRATEGY,
        'encoding_strategy': ENCODING_STRATEGY,
        'scaling_method': SCALING_METHOD,
        'handle_imbalance': HANDLE_IMBALANCE
    },
    'models': {
        'random_forest': RANDOM_FOREST_PARAMS,
        'xgboost': XGBOOST_PARAMS,
        'logistic_regression': LOGISTIC_REGRESSION_PARAMS,
        'svm': SVM_PARAMS
    },
    'training': {
        'cv_folds': CV_FOLDS,
        'enable_tuning': ENABLE_HYPERPARAMETER_TUNING,
        'scoring_metric': SCORING_METRIC
    },
    'visualization': {
        'plot_style': PLOT_STYLE,
        'figure_size': FIGURE_SIZE,
        'plot_dpi': PLOT_DPI
    },
    'app': {
        'title': APP_TITLE,
        'icon': APP_ICON,
        'layout': PAGE_LAYOUT
    }
}


# Run validation when imported
if __name__ == "__main__":
    validate_config()
    print("\nConfiguration Summary:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Target Column: {TARGET_COLUMN}")
    print(f"  Test Size: {TEST_SIZE}")
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Plots Directory: {PLOTS_DIR}")
