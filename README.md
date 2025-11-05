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

## Methodology (TO DO)

### 1. Data Preprocessing
- Imputed missing values (`person_emp_length`, `loan_int_rate`)
- Encoded categorical variables (One-Hot Encoding / Ordinal Encoding)
- Scaled numerical features using `StandardScaler`
- Split data into training and test sets (80/20)

### 2. Exploratory Data Analysis (EDA)
- Distribution and correlation analysis
- Default rate by loan grade and intent
- Feature relationships with income and credit history

### 3. Model Development
- **Baseline models:** Logistic Regression, Decision Tree  
- **Advanced models:** Random Forest, XGBoost, Gradient Boosting  
- Hyperparameter tuning with GridSearchCV and cross-validation

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Model interpretability using Feature Importance and SHAP analysis

---

## Results (TO DO)
- The top-performing model achieved a high **ROC-AUC**, balancing precision and recall.  
- Key predictors of default included:
  - `loan_grade`
  - `loan_percent_income`
  - `cb_person_default_on_file`
  - `loan_int_rate`
- Feature importance analysis confirmed that both **borrower credit history** and **loan characteristics** are crucial indicators of risk.

---

## Repository Structure (TO DO)

```bash
├── app
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── dist (or build)
├── node_modules
├── bower_components (if using bower)
├── test
├── Gruntfile.js/gulpfile.js
├── README.md
├── package.json
├── bower.json (if using bower)
└── .gitignore
```

---

## Requirements & Dependencies (TO DO)
**Python Version:** 3.9 or above  

Install all dependencies:

```bash
pip install -r requirements.txt
```

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

## Acknowledgments (TO DO)

- Dataset: [Kaggle – Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Libraries & Tools: Python, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
- Inspiration: Financial risk analytics and applied data science research in credit scoring.
