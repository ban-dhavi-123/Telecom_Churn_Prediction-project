"""
Exploratory Data Analysis and Visualization Script
==================================================
This script performs comprehensive EDA including:
- Data distribution analysis
- Correlation analysis
- Feature importance visualization
- Churn analysis by different features
- Statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ChurnEDA:
    """
    Comprehensive EDA class for telecom churn analysis.
    """
    
    def __init__(self, data_path):
        """
        Initialize EDA with data path.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset
        """
        self.data_path = data_path
        self.df = None
        self.plots_dir = '../plots'
        
    def load_data(self):
        """
        Load the dataset.
        
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        print("="*60)
        print("LOADING DATA FOR EDA")
        print("="*60)
        
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"\nData loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def basic_info(self):
        """
        Display basic information about the dataset.
        """
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1]}")
        print(f"Number of Records: {self.df.shape[0]}")
        
        print("\n" + "-"*60)
        print("Data Types:")
        print("-"*60)
        print(self.df.dtypes)
        
        print("\n" + "-"*60)
        print("Missing Values:")
        print("-"*60)
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        print("\n" + "-"*60)
        print("Statistical Summary:")
        print("-"*60)
        print(self.df.describe())
    
    def plot_target_distribution(self, target_col='Churn'):
        """
        Plot the distribution of the target variable.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        """
        if target_col not in self.df.columns:
            print(f"Warning: Column '{target_col}' not found in dataset")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        churn_counts = self.df[target_col].value_counts()
        axes[0].bar(churn_counts.index.astype(str), churn_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title(f'{target_col} Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Target distribution plot saved to: {self.plots_dir}/target_distribution.png")
        plt.close()
    
    def plot_numerical_distributions(self):
        """
        Plot distributions of all numerical features.
        """
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_cols:
            print("No numerical columns found")
            return
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/numerical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Numerical distributions plot saved to: {self.plots_dir}/numerical_distributions.png")
        plt.close()
    
    def plot_categorical_distributions(self, target_col='Churn'):
        """
        Plot distributions of categorical features with churn analysis.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        if not categorical_cols:
            print("No categorical columns found")
            return
        
        n_cols = 2
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(categorical_cols):
            if target_col in self.df.columns:
                # Create grouped bar chart
                churn_data = self.df.groupby([col, target_col]).size().unstack(fill_value=0)
                churn_data.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'])
                axes[idx].set_title(f'{col} vs {target_col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].legend(title=target_col)
                axes[idx].grid(axis='y', alpha=0.3)
                plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # Simple count plot
                self.df[col].value_counts().plot(kind='bar', ax=axes[idx], color='skyblue')
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Categorical distributions plot saved to: {self.plots_dir}/categorical_distributions.png")
        plt.close()
    
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix for numerical features.
        """
        # Select only numerical columns
        numerical_df = self.df.select_dtypes(include=['int64', 'float64'])
        
        if numerical_df.empty:
            print("No numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to: {self.plots_dir}/correlation_matrix.png")
        plt.close()
    
    def plot_boxplots(self, target_col='Churn'):
        """
        Plot boxplots for numerical features grouped by target variable.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        """
        if target_col not in self.df.columns:
            print(f"Warning: Column '{target_col}' not found")
            return
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target if it's numerical
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if not numerical_cols:
            print("No numerical columns for boxplot analysis")
            return
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            self.df.boxplot(column=col, by=target_col, ax=axes[idx])
            axes[idx].set_title(f'{col} by {target_col}', fontweight='bold')
            axes[idx].set_xlabel(target_col)
            axes[idx].set_ylabel(col)
            plt.sca(axes[idx])
            plt.xticks(rotation=0)
        
        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/boxplots_by_churn.png', dpi=300, bbox_inches='tight')
        print(f"✓ Boxplots saved to: {self.plots_dir}/boxplots_by_churn.png")
        plt.close()
    
    def generate_summary_report(self, target_col='Churn'):
        """
        Generate a comprehensive summary report.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        """
        print("\n" + "="*70)
        print(" "*20 + "SUMMARY REPORT")
        print("="*70)
        
        print(f"\n1. Dataset Overview:")
        print(f"   - Total Records: {len(self.df)}")
        print(f"   - Total Features: {self.df.shape[1]}")
        print(f"   - Numerical Features: {len(self.df.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"   - Categorical Features: {len(self.df.select_dtypes(include=['object']).columns)}")
        
        if target_col in self.df.columns:
            print(f"\n2. Target Variable ({target_col}) Distribution:")
            churn_counts = self.df[target_col].value_counts()
            for label, count in churn_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"   - {label}: {count} ({percentage:.2f}%)")
        
        print(f"\n3. Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   - No missing values found!")
        else:
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.df)) * 100
                print(f"   - {col}: {count} ({percentage:.2f}%)")
        
        print(f"\n4. Duplicate Records:")
        duplicates = self.df.duplicated().sum()
        print(f"   - {duplicates} duplicate records found")
        
        print("\n" + "="*70)
    
    def run_full_eda(self, target_col='Churn'):
        """
        Run complete EDA pipeline.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        """
        import os
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(" "*15 + "EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Basic info
        self.basic_info()
        
        # Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.plot_target_distribution(target_col)
        self.plot_numerical_distributions()
        self.plot_categorical_distributions(target_col)
        self.plot_correlation_matrix()
        self.plot_boxplots(target_col)
        
        # Generate summary report
        self.generate_summary_report(target_col)
        
        print("\n" + "="*70)
        print(" "*20 + "EDA COMPLETED!")
        print("="*70)
        print(f"\nAll plots saved to: {self.plots_dir}/")


# Example usage
if __name__ == "__main__":
    # Initialize EDA
    # Replace with actual path: r"C:\Users\lssan\Downloads\P585 Churn.xlsx"
    data_path = "../data/telecom_churn.csv"  # or .xlsx
    
    eda = ChurnEDA(data_path)
    
    # Run full EDA
    eda.run_full_eda(target_col='Churn')  # Adjust target column name as needed
    
    print("\nEDA completed successfully!")
