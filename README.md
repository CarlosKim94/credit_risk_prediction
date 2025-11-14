# Credit Risk Prediction using Machine Learning

## Overview
This project develops machine learning models to predict the likelihood of loan default using a dataset of over 32,000 loan applications.  
By analyzing borrower demographics, employment information, loan characteristics, and credit history, the model identifies patterns that differentiate reliable borrowers from high-risk applicants.

Accurately predicting loan defaults enables financial institutions to make better credit decisions and reduce risk exposure.

---

## Dataset
The dataset is sourced from **Kaggle – Credit Risk Dataset**.  
It contains **12 features** and **32,581 records** describing borrower profiles and loan details.

| Feature | Description |
|----------|-------------|
| `person_age` | Age of the loan applicant |
| `person_income` | Annual income of the applicant |
| `person_home_ownership` | Type of home ownership (rent, mortgage, own, other) |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan (education, personal, home improvement, etc.) |
| `loan_grade` | Credit grade (A-G) reflecting creditworthiness |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate of the loan |
| `loan_percent_income` | Ratio of loan amount to income |
| `cb_person_default_on_file` | Credit bureau record of past defaults (Y/N) |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `loan_status` | **Target variable** - 0: non-default, 1: default |

---

## Objective
The goal is to **predict loan default risk** based on borrower and loan features using machine learning models.  
This project compares multiple machine learning algorithms and evaluates their performance and interpretability.

---

## Methodology

### 1. Data Preprocessing
- Imputed missing values (`person_emp_length`, `loan_int_rate`)
- Encoded categorical variables (DictVectorizer)
- Split data into training, validation and test sets (60/20/20)

### 2. Exploratory Data Analysis (EDA)
- Distribution and correlation of all features
  ![Distribution and Correlation of All Features](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/correlation.png)
  
- Correlation Heatmap of all features
  
  ![Correlation Heatmap of All Features](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/correlation_heatmap.png)
  
- Distribution of the target variable (`loan_satus`)
  
  ![Distribution of loan_status](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/loan_status_distribution.png)
  
- Feature relationships with loan intent and loan status
  
  ![Relationship of loan intent and loan_status](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/loan_intent.png)

### 3. Model Development
- **Baseline models:** Logistic Regression, Random Forest 
- Hyperparameter tuning with different ranges of estimators and depths

### 4. Evaluation
- Metrics: ROC-AUC

---

## Results
- The top-performing model achieved a high **ROC-AUC**, balancing precision and recall.  
- Key findings:
  - Base Logistic Regression model ROC_AUC score on validation set is 0.7617
  - Different values of C did not show any improvement
  - Base Logistic Regression model ROC_AUC score on set set is 0.7545
  - Base Random Forest model ROC_AUC score on validation set is 0.8389
    ![result](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/EDA/result.png)
  - After hyper parameter tuning, the best performing Random Forest model is max_depth = 25 and n_estimator = 60, and the model is saved as 'model_depth_25_estimator_60_0.858.bin'
  - Fine-tuned Random Forest model ROC_AUC score on test set is 0.8525

---

## Repository Structure

```bash
├── EDA
│   ├── correlation.png
│   ├── correlation_heatmap.png
│   ├── loan_intent.png
│   ├── loan_status_distribution.png
│   └── result.png
├── data
│   └── credit_risk_dataset.csv
├── model
│   ├── model_depth_10_estimator_10_0.839.bin
│   ├── model_depth_10_estimator_20_0.845.bin
│   ├── model_depth_10_estimator_40_0.846.bin
│   ├── model_depth_15_estimator_100_0.855.bin
│   ├── model_depth_15_estimator_10_0.851.bin
│   ├── model_depth_15_estimator_20_0.853.bin
│   ├── model_depth_15_estimator_60_0.854.bin
│   ├── model_depth_20_estimator_20_0.857.bin
│   └── model_depth_25_estimator_60_0.858.bin
├── Dockerfile
├── README.md
├── client01.py
├── credit_risk_prediction.ipynb
├── model_training.py
├── predict.py
├── pyproject.toml
└── uv.lock
```

Jupyter Notebook for EDA, data preprocessing, model training, hyper parameter tuning
- [credit_risk_prediction.ipynb](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/credit_risk_prediction.ipynb)

Python script for data pre-processing and training
- [model_training.py](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/model_training.py)
  
---

## Requirements & Dependencies
**Python Version:** 3.12 or above  

[Dependencies](https://github.com/CarlosKim94/credit_risk_prediction/blob/main/pyproject.toml):
- fastapi>=0.121.1
- matplotlib>=3.10.7
- numpy>=2.3.4
- pandas>=2.3.3
- requests>=2.32.5
- scikit-learn>=1.7.2
- seaborn>=0.13.2
- uvicorn>=0.38.0

Dependencies will all be automatically installed while deploying in the Docker container in the following section

---
## How to Run the Project (TO DO)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/Loan_Default_Prediction.ipynb
```

### 4. Reproduce Results

- Execute all cells in order
- Review plots, metrics, and feature importance
- Compare algorithm performance in the results/ folder

---

## Acknowledgments

- Dataset: [Kaggle – Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Libraries & Tools: Python, scikit-learn, Pandas, Seaborn, Numpy, fastAPI, uvicorn, Docker
- Inspiration: Financial risk analytics and applied data science research in credit scoring.
