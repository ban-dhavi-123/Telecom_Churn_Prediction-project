"""
Data Preprocessing Script for Telecom Customer Churn Prediction
================================================================
This script handles:
- Loading data from Excel/CSV files
- Data cleaning (handling missing values, duplicates)
- Feature encoding (Label Encoding, One-Hot Encoding)
- Feature scaling (StandardScaler)
- Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for telecom churn prediction.
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset (CSV or Excel file)
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load data from CSV or Excel file.
        
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        print(f"Loading data from: {self.data_path}")
        
        # Check file extension and load accordingly
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
        
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def explore_data(self):
        """
        Display basic information about the dataset.
        """
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1]}")
        print(f"Number of Samples: {self.df.shape[0]}")
        
        print("\n" + "-"*60)
        print("COLUMN INFORMATION")
        print("-"*60)
        print(self.df.info())
        
        print("\n" + "-"*60)
        print("FIRST 5 ROWS")
        print("-"*60)
        print(self.df.head())
        
        print("\n" + "-"*60)
        print("STATISTICAL SUMMARY")
        print("-"*60)
        print(self.df.describe())
        
        print("\n" + "-"*60)
        print("MISSING VALUES")
        print("-"*60)
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
    def clean_data(self):
        """
        Clean the dataset by handling missing values and duplicates.
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataset
        """
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows found: {duplicates}")
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows.")
        
        # Handle missing values
        print("\nHandling missing values...")
        
        # For numerical columns: fill with median
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                print(f"  - Filled {col} with median: {median_value}")
        
        # For categorical columns: fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                print(f"  - Filled {col} with mode: {mode_value}")
        
        print(f"\nCleaned data shape: {self.df.shape}")
        return self.df
    
    def encode_features(self, target_column='Churn'):
        """
        Encode categorical features using Label Encoding and One-Hot Encoding.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column (default: 'Churn')
        
        Returns:
        --------
        pd.DataFrame : Encoded dataset
        """
        print("\n" + "="*60)
        print("FEATURE ENCODING")
        print("="*60)
        
        # Identify categorical columns (excluding target)
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from categorical list if present
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        print(f"\nCategorical columns to encode: {categorical_cols}")
        
        # Apply Label Encoding for binary categorical features
        binary_cols = [col for col in categorical_cols 
                      if self.df[col].nunique() == 2]
        
        for col in binary_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  - Label encoded: {col}")
        
        # Apply One-Hot Encoding for multi-class categorical features
        multi_class_cols = [col for col in categorical_cols 
                           if self.df[col].nunique() > 2]
        
        if multi_class_cols:
            print(f"\nApplying One-Hot Encoding to: {multi_class_cols}")
            self.df = pd.get_dummies(self.df, columns=multi_class_cols, drop_first=True)
        
        # Encode target variable if it's categorical
        if target_column in self.df.columns:
            if self.df[target_column].dtype == 'object':
                le_target = LabelEncoder()
                self.df[target_column] = le_target.fit_transform(self.df[target_column])
                self.label_encoders[target_column] = le_target
                print(f"\n  - Target variable '{target_column}' encoded")
                print(f"    Classes: {le_target.classes_}")
        
        print(f"\nEncoded data shape: {self.df.shape}")
        return self.df
    
    def split_data(self, target_column='Churn', test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column
        test_size : float
            Proportion of dataset to include in test split (default: 0.2)
        random_state : int
            Random state for reproducibility (default: 42)
        
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT")
        print("="*60)
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        # Display class distribution
        print("\nClass distribution in training set:")
        print(self.y_train.value_counts())
        print("\nClass distribution in testing set:")
        print(self.y_test.value_counts())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale numerical features using StandardScaler.
        
        Returns:
        --------
        tuple : Scaled X_train, X_test
        """
        print("\n" + "="*60)
        print("FEATURE SCALING")
        print("="*60)
        
        # Fit scaler on training data and transform both train and test
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Features scaled successfully using StandardScaler")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test
    
    def save_preprocessed_data(self, output_dir='../models'):
        """
        Save preprocessed data and preprocessing objects.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the preprocessed data and objects
        """
        print("\n" + "="*60)
        print("SAVING PREPROCESSED DATA")
        print("="*60)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train-test splits
        joblib.dump(self.X_train, os.path.join(output_dir, 'X_train.pkl'))
        joblib.dump(self.X_test, os.path.join(output_dir, 'X_test.pkl'))
        joblib.dump(self.y_train, os.path.join(output_dir, 'y_train.pkl'))
        joblib.dump(self.y_test, os.path.join(output_dir, 'y_test.pkl'))
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        
        print(f"Preprocessed data saved to: {output_dir}")
        print("  - X_train.pkl")
        print("  - X_test.pkl")
        print("  - y_train.pkl")
        print("  - y_test.pkl")
        print("  - scaler.pkl")
        print("  - label_encoders.pkl")
    
    def run_full_pipeline(self, target_column='Churn', test_size=0.2, random_state=42):
        """
        Run the complete preprocessing pipeline.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random state for reproducibility
        
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        print("\n" + "="*70)
        print(" "*15 + "DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Clean data
        self.clean_data()
        
        # Step 4: Encode features
        self.encode_features(target_column=target_column)
        
        # Step 5: Split data
        self.split_data(target_column=target_column, test_size=test_size, random_state=random_state)
        
        # Step 6: Scale features
        self.scale_features()
        
        # Step 7: Save preprocessed data
        self.save_preprocessed_data()
        
        print("\n" + "="*70)
        print(" "*20 + "PREPROCESSING COMPLETED!")
        print("="*70)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor with your data path
    # Replace with actual path: r"C:\Users\lssan\Downloads\P585 Churn.xlsx"
    data_path = "../data/telecom_churn.csv"  # or .xlsx
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor(data_path)
    
    # Run full preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline(
        target_column='Churn',  # Adjust based on your dataset
        test_size=0.2,
        random_state=42
    )
    
    print("\nPreprocessing completed successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
