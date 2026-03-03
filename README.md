#  Explainable AI-Based Loan Default Prediction

##  Overview
This project builds an end-to-end, leakage-free credit risk prediction system using the Lending Club dataset (2007–2015).  
The system predicts loan default probability using XGBoost and provides model interpretability using SHAP (Explainable AI).

---

##  Problem Statement
Predict whether a borrower will default on a loan using financial and credit-related features while ensuring:
- Removal of post-outcome data leakage
- Proper handling of class imbalance
- Model interpretability

---

##  Approach

### Data Preprocessing
- Filtered loan status to Fully Paid and Charged Off
- Removed post-loan leakage features
- Dropped high-missing columns (>50%)
- Imputed missing values (Median/Mode)
- Encoded categorical features

### Class Imbalance Handling
- Logistic Regression with class_weight
- XGBoost with scale_pos_weight

### Models Compared
- Logistic Regression (baseline)
- Random Forest
- XGBoost (final model)

---

## 📊 Final Model Performance (Leakage-Free)

- ROC-AUC: ~0.71
- Default Recall: ~0.60
- F1-score (Default): ~0.44

---

## Explainable AI (SHAP)

SHAP analysis identified the key drivers of loan default:

- Loan sub-grade
- Interest rate
- Debt-to-income ratio (DTI)
- Loan term
- Annual income
- Loan amount

These results align with financial risk theory.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib

---

## 📂 Dataset
Dataset used: Lending Club Loan Data (2007–2015)  
Available on Kaggle.

(Note: Dataset not uploaded due to size constraints.)

---

##  Key Learning Outcomes
- Handling real-world class imbalance
- Detecting and removing data leakage
- Model comparison and evaluation
- Threshold tuning for risk optimization
- Explainable AI using SHAP
