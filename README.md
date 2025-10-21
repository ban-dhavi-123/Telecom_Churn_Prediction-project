# 📱 Telecom Customer Churn Prediction using Machine Learning

A comprehensive end-to-end machine learning project for predicting customer churn in the telecommunications industry. This project includes data preprocessing, exploratory data analysis, model training with multiple algorithms, and a Streamlit-based web application for deployment.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Docker Deployment](#docker-deployment)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

Customer churn is a critical metric for telecommunications companies. This project aims to predict which customers are likely to churn (leave the service) using machine learning techniques. By identifying at-risk customers early, companies can take proactive measures to retain them.

### Objectives:
- Perform comprehensive exploratory data analysis (EDA)
- Build and compare multiple machine learning models
- Create an interactive web application for predictions
- Provide actionable insights for customer retention

---

## ✨ Features

- **Data Preprocessing**: Automated data cleaning, encoding, and scaling
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Random Forest, XGBoost, Logistic Regression, SVM
- **Model Comparison**: Detailed performance metrics and visualizations
- **Interactive Web App**: Streamlit-based UI for single and batch predictions
- **Docker Support**: Containerized deployment for easy scaling
- **Model Persistence**: Save and load trained models
- **Comprehensive Documentation**: Well-commented code and detailed README

---

## 📁 Project Structure

```
Telecom_Churn_Prediction/
│
├── data/                          # Dataset directory
│   └── telecom_churn.csv         # Your dataset (place here)
│
├── notebooks/                     # Jupyter notebooks
│   └── telecom_churn_eda.ipynb   # EDA notebook
│
├── scripts/                       # Python scripts
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── model_training.py         # Model training and evaluation
│   └── eda_visualization.py      # EDA and visualization script
│
├── models/                        # Saved models directory
│   ├── best_model.pkl            # Best performing model
│   ├── random_forest.pkl         # Random Forest model
│   ├── xgboost.pkl               # XGBoost model
│   ├── logistic_regression.pkl   # Logistic Regression model
│   ├── support_vector_machine.pkl # SVM model
│   ├── scaler.pkl                # Feature scaler
│   └── label_encoders.pkl        # Label encoders
│
├── plots/                         # Generated visualizations
│   ├── confusion_matrices.png    # Confusion matrices
│   ├── roc_curves.png            # ROC curves
│   └── metrics_comparison.png    # Model metrics comparison
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── README.md                      # Project documentation
└── project_summary.txt            # Project summary
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Telecom_Churn_Prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Your Dataset

1. Place your dataset in the `data/` folder
2. Supported formats: CSV (`.csv`) or Excel (`.xlsx`, `.xls`)
3. Example: `data/telecom_churn.csv`

**For this project, place your dataset:**
```
C:\Users\lssan\Downloads\P585 Churn.xlsx → data/telecom_churn.xlsx
```

---

## 📊 Usage

### 1. Exploratory Data Analysis (EDA)

#### Option A: Using Jupyter Notebook
```bash
jupyter notebook notebooks/telecom_churn_eda.ipynb
```

#### Option B: Using Python Script
```bash
python scripts/eda_visualization.py
```

**Note**: Update the `data_path` variable in the script to point to your dataset.

### 2. Data Preprocessing

```bash
cd scripts
python data_preprocessing.py
```

This will:
- Load and clean the data
- Handle missing values
- Encode categorical features
- Scale numerical features
- Split data into train/test sets
- Save preprocessed data to `models/` directory

### 3. Model Training

```bash
python scripts/model_training.py
```

This will:
- Train multiple ML models (Random Forest, XGBoost, Logistic Regression, SVM)
- Evaluate and compare models
- Generate performance visualizations
- Save all models to `models/` directory
- Save plots to `plots/` directory

### 4. Run Streamlit Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### App Features:
- **Single Prediction**: Enter customer details for individual predictions
- **Batch Prediction**: Upload CSV/Excel file for bulk predictions
- **Model Performance**: View model metrics and visualizations

---

## 🤖 Models

### Implemented Algorithms:

1. **Random Forest Classifier**
   - Ensemble learning method
   - Robust to overfitting
   - Handles non-linear relationships

2. **XGBoost Classifier**
   - Gradient boosting algorithm
   - High performance and speed
   - Excellent for structured data

3. **Logistic Regression**
   - Linear classification model
   - Interpretable coefficients
   - Fast training and prediction

4. **Support Vector Machine (SVM)**
   - Finds optimal decision boundary
   - Effective in high-dimensional spaces
   - Kernel trick for non-linear patterns

### Evaluation Metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: Correct positive predictions ratio
- **Recall**: Actual positives identified ratio
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's discrimination ability

---

## 📈 Results

After training, you'll find:

### Model Performance Files:
- `plots/confusion_matrices.png` - Confusion matrices for all models
- `plots/roc_curves.png` - ROC curves comparison
- `plots/metrics_comparison.png` - Bar chart of all metrics

### Saved Models:
- `models/best_model.pkl` - Best performing model
- `models/random_forest.pkl` - Random Forest model
- `models/xgboost.pkl` - XGBoost model
- `models/logistic_regression.pkl` - Logistic Regression model
- `models/support_vector_machine.pkl` - SVM model

### Expected Performance:
- Accuracy: 75-85%
- ROC-AUC: 0.80-0.90
- F1-Score: 0.70-0.80

*(Actual results depend on your dataset)*

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t telecom-churn-prediction .
```

### Run Docker Container

```bash
docker run -p 8501:8501 telecom-churn-prediction
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

### Docker Commands:

```bash
# Stop container
docker stop <container-id>

# Remove container
docker rm <container-id>

# Remove image
docker rmi telecom-churn-prediction

# View running containers
docker ps

# View logs
docker logs <container-id>
```

---

## 🛠️ Technologies Used

### Core Libraries:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting framework

### Visualization:
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualizations
- **plotly** - Interactive visualizations

### Web Framework:
- **streamlit** - Web application framework

### Model Persistence:
- **joblib** - Model serialization

### Additional:
- **openpyxl** - Excel file support
- **imbalanced-learn** - Handling class imbalance
- **jupyter** - Interactive notebooks

---

## 📝 Configuration

### Update Dataset Path

In each script, update the `data_path` variable:

```python
# For CSV
data_path = "data/telecom_churn.csv"

# For Excel
data_path = r"C:\Users\lssan\Downloads\P585 Churn.xlsx"
```

### Update Target Column

If your target column has a different name, update:

```python
target_column = 'Churn'  # Change to your column name
```

### Adjust Model Parameters

In `scripts/model_training.py`, you can modify model hyperparameters:

```python
RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum depth
    random_state=42
)
```

---

## 🔧 Troubleshooting

### Common Issues:

1. **File Not Found Error**
   - Ensure dataset is in the `data/` folder
   - Check file path and extension

2. **Module Not Found Error**
   - Install missing packages: `pip install <package-name>`
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Model Not Loaded in Streamlit**
   - Run preprocessing script first
   - Run training script to generate models
   - Ensure models are saved in `models/` directory

4. **Memory Error**
   - Reduce dataset size
   - Adjust model parameters (e.g., reduce `n_estimators`)

---

## 📚 Additional Resources

### Documentation:
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)

### Tutorials:
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Towards Data Science](https://towardsdatascience.com/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Your Name** - *Initial work*

---

## 🙏 Acknowledgments

- Dataset source: [Specify your data source]
- Inspiration: Telecom industry churn analysis
- Libraries: Thanks to all open-source contributors

---

## 📞 Contact

For questions or feedback:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🔄 Version History

- **v1.0.0** (2024-10-21)
  - Initial release
  - Complete ML pipeline
  - Streamlit web application
  - Docker support

---

## 📊 Project Status

**Status**: ✅ Complete and Production-Ready

**Last Updated**: October 21, 2024

---

**⭐ If you find this project helpful, please consider giving it a star!**
