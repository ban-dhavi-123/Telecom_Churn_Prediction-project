# Data Directory

## 📁 Dataset Placement

Place your telecom churn dataset in this directory.

### Supported Formats:
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

### Your Dataset:

**Original Location:**
```
C:\Users\lssan\Downloads\P585 Churn.xlsx
```

**Copy to:**
```
data/telecom_churn.xlsx
```

Or rename it to:
```
data/telecom_churn.csv (if converted to CSV)
```

---

## 📊 Expected Dataset Structure

Your dataset should contain the following types of columns:

### Customer Information:
- Customer ID (optional, will be excluded from modeling)
- Demographics (age, gender, etc.)

### Account Information:
- Tenure (months with the company)
- Contract type (Month-to-month, One year, Two year)
- Payment method

### Services:
- Phone service
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming TV
- Streaming movies

### Billing:
- Monthly charges
- Total charges
- Paperless billing

### Target Variable:
- **Churn** (Yes/No or 1/0) - This is what we're predicting

---

## 🔄 Data Loading Instructions

### Option 1: Copy File Here

1. Copy your file from:
   ```
   C:\Users\lssan\Downloads\P585 Churn.xlsx
   ```

2. Paste it into this `data/` folder

3. Rename to `telecom_churn.xlsx` (or keep original name and update scripts)

### Option 2: Update Path in Scripts

If you prefer to keep the file in its original location, update the `data_path` variable in:

- `scripts/data_preprocessing.py`
- `scripts/eda_visualization.py`
- `notebooks/telecom_churn_eda.ipynb`

Example:
```python
data_path = r"C:\Users\lssan\Downloads\P585 Churn.xlsx"
```

---

## ✅ Verification

After placing your dataset, verify it loads correctly:

```python
import pandas as pd

# For CSV
df = pd.read_csv('data/telecom_churn.csv')

# For Excel
df = pd.read_excel('data/telecom_churn.xlsx')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
```

---

## 📝 Notes

- The `data/` folder is included in `.gitignore` to prevent accidentally committing large datasets
- Make sure your dataset has a clear target column (e.g., 'Churn')
- Check for any special characters or encoding issues in column names
- Ensure there are no completely empty rows or columns

---

## 🚨 Important

**Do not commit sensitive customer data to version control!**

The `.gitignore` file is configured to exclude data files, but always verify before pushing to a repository.
