"""
Complete Pipeline Runner for Telecom Churn Prediction
======================================================
This script runs the entire ML pipeline from data preprocessing to model training.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(step_num, text):
    """Print step information."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {text}")
    print(f"{'='*70}\n")


def check_data_exists():
    """Check if dataset exists."""
    data_dir = Path("data")
    
    # Look for CSV or Excel files
    csv_files = list(data_dir.glob("*.csv"))
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    all_files = csv_files + excel_files
    
    if not all_files:
        print("❌ No dataset found in data/ directory!")
        print("\nPlease place your dataset in the data/ folder.")
        print("Supported formats: .csv, .xlsx, .xls")
        print("\nExample:")
        print("  Copy: C:\\Users\\lssan\\Downloads\\P585 Churn.xlsx")
        print("  To:   data/telecom_churn.xlsx")
        return None
    
    print(f"✓ Found {len(all_files)} dataset(s):")
    for idx, file in enumerate(all_files, 1):
        print(f"  {idx}. {file.name}")
    
    if len(all_files) == 1:
        return str(all_files[0])
    else:
        print("\nMultiple datasets found. Using the first one.")
        return str(all_files[0])


def run_preprocessing(data_path):
    """Run data preprocessing."""
    print_step(1, "DATA PREPROCESSING")
    
    try:
        from scripts.data_preprocessing import DataPreprocessor
        
        # Update data path in the script
        preprocessor = DataPreprocessor(data_path)
        
        # Run full pipeline
        X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline(
            target_column='Churn',  # Adjust if needed
            test_size=0.2,
            random_state=42
        )
        
        print("\n✓ Preprocessing completed successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {str(e)}")
        print("\nPlease check:")
        print("  1. Dataset format is correct")
        print("  2. Target column name (default: 'Churn')")
        print("  3. No corrupted data")
        return False


def run_training():
    """Run model training."""
    print_step(2, "MODEL TRAINING")
    
    try:
        from scripts.model_training import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer(models_dir='models')
        
        # Run full training pipeline
        models, results_df, best_model = trainer.run_full_pipeline()
        
        print("\n✓ Model training completed successfully!")
        print(f"\nBest Model: {best_model}")
        print("\nModel Performance:")
        print(results_df.to_string(index=False))
        
        return True
    
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("\nPlease check:")
        print("  1. Preprocessing completed successfully")
        print("  2. Preprocessed data exists in models/ directory")
        return False


def run_streamlit_app():
    """Launch Streamlit application."""
    print_step(3, "LAUNCHING STREAMLIT APP")
    
    print("Starting Streamlit application...")
    print("The app will open in your default browser.")
    print("\nPress Ctrl+C to stop the server.\n")
    
    time.sleep(2)
    
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nStreamlit app stopped.")
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {str(e)}")
        print("\nTry running manually: streamlit run app.py")


def main():
    """Main pipeline execution."""
    
    print_header("TELECOM CHURN PREDICTION - COMPLETE PIPELINE")
    
    print("""
    This script will:
    1. Preprocess your data
    2. Train multiple ML models
    3. Launch the Streamlit web application
    
    Make sure your dataset is in the data/ folder!
    """)
    
    # Check if data exists
    print_header("CHECKING DATASET")
    data_path = check_data_exists()
    
    if not data_path:
        return
    
    print(f"\nUsing dataset: {data_path}")
    
    # Ask user to confirm
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nPipeline cancelled.")
        return
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Step 1: Preprocessing
    if not run_preprocessing(data_path):
        print("\n❌ Pipeline stopped due to preprocessing error.")
        return
    
    # Step 2: Training
    if not run_training():
        print("\n❌ Pipeline stopped due to training error.")
        return
    
    # Step 3: Launch app
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    
    print("""
    ✓ Data preprocessed
    ✓ Models trained
    ✓ Ready for deployment
    
    Next steps:
    1. Launch Streamlit app (Option 1)
    2. View results in plots/ directory (Option 2)
    3. Exit (Option 3)
    """)
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        run_streamlit_app()
    elif choice == '2':
        print("\nPlots saved in: plots/")
        print("Models saved in: models/")
        print("\nTo view plots, open the files in plots/ directory")
    else:
        print("\nThank you for using the Telecom Churn Prediction system!")
        print("To launch the app later, run: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\nPlease check the error message and try again.")
