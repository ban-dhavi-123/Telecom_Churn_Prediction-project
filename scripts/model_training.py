"""
Model Training Script for Telecom Customer Churn Prediction
============================================================
This script handles:
- Loading preprocessed data
- Training multiple ML models (Random Forest, XGBoost, Logistic Regression, SVM)
- Model evaluation and comparison
- Hyperparameter tuning
- Model saving
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A comprehensive model training class for churn prediction.
    """
    
    def __init__(self, models_dir='../models'):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing preprocessed data and for saving models
        """
        self.models_dir = models_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_preprocessed_data(self):
        """
        Load preprocessed training and testing data.
        
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        print("="*60)
        print("LOADING PREPROCESSED DATA")
        print("="*60)
        
        # Load train-test splits
        self.X_train = joblib.load(os.path.join(self.models_dir, 'X_train.pkl'))
        self.X_test = joblib.load(os.path.join(self.models_dir, 'X_test.pkl'))
        self.y_train = joblib.load(os.path.join(self.models_dir, 'y_train.pkl'))
        self.y_test = joblib.load(os.path.join(self.models_dir, 'y_test.pkl'))
        
        print(f"\nData loaded successfully!")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")
        print(f"Target distribution in training set:")
        print(pd.Series(self.y_train).value_counts())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """
        Initialize all machine learning models.
        
        Returns:
        --------
        dict : Dictionary of initialized models
        """
        print("\n" + "="*60)
        print("INITIALIZING MODELS")
        print("="*60)
        
        # Define models with default parameters
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
        }
        
        print("\nModels initialized:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_models(self):
        """
        Train all initialized models.
        
        Returns:
        --------
        dict : Dictionary of trained models
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  ✓ {name} trained successfully!")
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)
        
        return self.models
    
    def evaluate_model(self, model_name, model):
        """
        Evaluate a single model and store results.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn model
            Trained model object
        
        Returns:
        --------
        dict : Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store results
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Predictions': y_pred,
            'Probabilities': y_pred_proba
        }
        
        return results
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models and compare performance.
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing all model results
        """
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            results = self.evaluate_model(name, model)
            self.results[name] = results
            
            # Store metrics for comparison
            results_list.append({
                'Model': name,
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score'],
                'ROC-AUC': results['ROC-AUC']
            })
            
            print(f"  Accuracy:  {results['Accuracy']:.4f}")
            print(f"  Precision: {results['Precision']:.4f}")
            print(f"  Recall:    {results['Recall']:.4f}")
            print(f"  F1-Score:  {results['F1-Score']:.4f}")
            print(f"  ROC-AUC:   {results['ROC-AUC']:.4f}")
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self, save_path='../plots'):
        """
        Plot confusion matrices for all models.
        
        Parameters:
        -----------
        save_path : str
            Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, results['Predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}\nAccuracy: {results["Accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to: {save_path}/confusion_matrices.png")
        plt.close()
    
    def plot_roc_curves(self, save_path='../plots'):
        """
        Plot ROC curves for all models.
        
        Parameters:
        -----------
        save_path : str
            Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['Probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['ROC-AUC']:.4f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}/roc_curves.png")
        plt.close()
    
    def plot_metrics_comparison(self, results_df, save_path='../plots'):
        """
        Plot bar chart comparing all metrics across models.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame containing model results
        save_path : str
            Directory to save plots
        """
        os.makedirs(save_path, exist_ok=True)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(results_df))
        width = 0.15
        
        for idx, metric in enumerate(metrics):
            offset = width * (idx - 2)
            ax.bar(x + offset, results_df[metric], width, label=metric)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}/metrics_comparison.png")
        plt.close()
    
    def save_best_model(self, results_df):
        """
        Save the best performing model.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame containing model results
        
        Returns:
        --------
        str : Name of the best model
        """
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        # Save the best model
        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        joblib.dump(best_model, model_path)
        
        print(f"\n" + "="*60)
        print(f"BEST MODEL: {best_model_name}")
        print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
        print(f"Saved to: {model_path}")
        print("="*60)
        
        return best_model_name
    
    def save_all_models(self):
        """
        Save all trained models.
        """
        print("\n" + "="*60)
        print("SAVING ALL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            # Create filename from model name
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(self.models_dir, filename)
            
            # Save model
            joblib.dump(model, filepath)
            print(f"  ✓ {name} saved to: {filename}")
        
        print("\nAll models saved successfully!")
    
    def run_full_pipeline(self):
        """
        Run the complete model training and evaluation pipeline.
        
        Returns:
        --------
        tuple : (models, results_df, best_model_name)
        """
        print("\n" + "="*70)
        print(" "*20 + "MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load preprocessed data
        self.load_preprocessed_data()
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Evaluate models
        results_df = self.evaluate_all_models()
        
        # Step 5: Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_metrics_comparison(results_df)
        
        # Step 6: Save models
        self.save_all_models()
        best_model_name = self.save_best_model(results_df)
        
        print("\n" + "="*70)
        print(" "*20 + "TRAINING COMPLETED!")
        print("="*70)
        
        return self.models, results_df, best_model_name


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer(models_dir='../models')
    
    # Run full training pipeline
    models, results_df, best_model = trainer.run_full_pipeline()
    
    print("\n" + "="*70)
    print("Training pipeline completed successfully!")
    print(f"Best Model: {best_model}")
    print("="*70)
