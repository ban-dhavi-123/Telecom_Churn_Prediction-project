# 🚀 Quick Start Guide

Get started with the Telecom Churn Prediction system in just a few minutes!

---

## ⚡ Fast Track (Automated)

### Windows Users:

1. **Place your dataset** in the `data/` folder:
   ```
   Copy: C:\Users\lssan\Downloads\P585 Churn.xlsx
   To:   data/telecom_churn.xlsx
   ```

2. **Double-click** `run_pipeline.bat`

3. **Wait** for the pipeline to complete (5-10 minutes)

4. **Access** the web app at `http://localhost:8501`

That's it! 🎉

---

## 📋 Step-by-Step (Manual)

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

Place your dataset in the `data/` folder:
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Example: `data/telecom_churn.xlsx`

### Step 3: Run Pipeline

```bash
# Option A: Automated pipeline
python run_pipeline.py

# Option B: Manual steps
cd scripts
python data_preprocessing.py
python model_training.py
cd ..
streamlit run app.py
```

### Step 4: Access Application

Open your browser and go to:
```
http://localhost:8501
```

---

## 🎯 What Each Step Does

### 1️⃣ Data Preprocessing
- Loads your dataset
- Cleans and handles missing values
- Encodes categorical features
- Scales numerical features
- Splits into train/test sets
- **Output**: Preprocessed data in `models/` folder

### 2️⃣ Model Training
- Trains 4 ML models:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine
- Evaluates and compares models
- Generates performance visualizations
- **Output**: Trained models in `models/` folder, plots in `plots/` folder

### 3️⃣ Web Application
- Interactive prediction interface
- Single customer prediction
- Batch prediction (upload file)
- Model performance dashboard
- **Access**: `http://localhost:8501`

---

## 📊 Using the Web App

### Single Prediction:
1. Select "Single Prediction" mode
2. Enter customer details
3. Click "Predict Churn"
4. View results and recommendations

### Batch Prediction:
1. Select "Batch Prediction" mode
2. Upload CSV/Excel file
3. Click "Generate Predictions"
4. Download results

### Model Performance:
1. Select "Model Performance" mode
2. View confusion matrices
3. Compare ROC curves
4. Analyze metrics

---

## 🐳 Docker Quick Start

```bash
# Build image
docker build -t telecom-churn-prediction .

# Run container
docker run -p 8501:8501 telecom-churn-prediction

# Access app
# Open: http://localhost:8501
```

---

## ⚠️ Troubleshooting

### Dataset Not Found
```
❌ Error: No dataset found in data/ directory
✅ Solution: Place your dataset in the data/ folder
```

### Module Not Found
```
❌ Error: ModuleNotFoundError: No module named 'pandas'
✅ Solution: pip install -r requirements.txt
```

### Model Not Loaded
```
❌ Error: Model file not found
✅ Solution: Run preprocessing and training first
```

### Port Already in Use
```
❌ Error: Port 8501 is already in use
✅ Solution: Stop other Streamlit apps or use different port:
   streamlit run app.py --server.port 8502
```

---

## 📁 Expected File Structure After Setup

```
Telecom_Churn_Prediction/
├── data/
│   └── telecom_churn.xlsx          ← Your dataset here
├── models/
│   ├── best_model.pkl              ← Generated after training
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── plots/
│   ├── confusion_matrices.png      ← Generated after training
│   ├── roc_curves.png
│   └── metrics_comparison.png
└── venv/                           ← Virtual environment
```

---

## 🎓 Next Steps

After completing the quick start:

1. **Explore the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/telecom_churn_eda.ipynb
   ```

2. **Customize Models**:
   - Edit `scripts/model_training.py`
   - Adjust hyperparameters
   - Add new models

3. **Improve Preprocessing**:
   - Edit `scripts/data_preprocessing.py`
   - Add feature engineering
   - Handle class imbalance

4. **Deploy to Production**:
   - Use Docker for deployment
   - Set up on cloud (AWS, Azure, GCP)
   - Add authentication

---

## 💡 Tips

- **First Time**: Run the automated pipeline (`run_pipeline.bat` or `run_pipeline.py`)
- **Experimentation**: Use Jupyter notebooks for exploration
- **Production**: Use Docker for deployment
- **Updates**: Retrain models periodically with new data

---

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review `project_summary.txt` for project overview
- Check error messages carefully
- Ensure all dependencies are installed

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Dataset placed in `data/` folder
- [ ] Preprocessing completed
- [ ] Models trained
- [ ] Web app running
- [ ] Predictions working

---

**🎉 Congratulations! You're ready to predict customer churn!**

For detailed documentation, see [README.md](README.md)
