# FraudGuard-AI
Machine learning based fraud detection system using Flask, Python, and Scikit-learn with OTP verification and case management.
# FraudGuard AI
## Intelligent Transaction Fraud Detection System

FraudGuard AI is a machine learning-based web application developed to detect suspicious and fraudulent transactions.  
The system uses a trained ML model along with rule-based checks to classify transactions, assign risk levels, and support fraud investigation workflow.

This project is built using **Python, Flask, Scikit-learn, Pandas, SQLite, HTML, and CSS**.

---

## Project Overview

With the rapid increase in online payments and digital transactions, fraud detection has become an important challenge for businesses and financial institutions.  
Manual fraud investigation is time-consuming and may fail to detect hidden fraud patterns.

FraudGuard AI solves this problem by:

- analyzing transaction data automatically
- predicting fraud probability
- classifying transactions into risk levels
- providing OTP verification for risky transactions
- maintaining investigation cases
- storing transaction history for monitoring and review

---

## Main Features

### 1. Manual Fraud Prediction
Users can manually enter transaction details and instantly get:
- fraud probability
- confidence level
- risk level
- action recommendation

### 2. CSV Upload
Users can upload a CSV file containing multiple transaction records for bulk fraud analysis.

### 3. Machine Learning Fraud Detection
A trained machine learning model analyzes transaction data and predicts suspicious behavior.

### 4. Fraud Probability Score
Each transaction is assigned a fraud score in percentage.

### 5. Risk Level Classification
Transactions are categorized into:
- Low Risk
- Medium Risk
- High Risk
- Very High Risk

### 6. OTP Verification
High-risk transactions can be verified using OTP.

### 7. Suspicious Transaction Ranking
The system ranks transactions by fraud probability.

### 8. Case Management
Suspicious transactions are automatically converted into cases and assigned to analysts.

### 9. Notifications
The system generates alerts for high-risk activities.

### 10. Customer Risk Profiles
The application shows customer-wise transaction risk summaries.

### 11. History Tracking
All processed transactions are stored and can be searched or reviewed later.

### 12. Fraud Simulation
Users can simulate transactions and test fraud behavior without saving the record permanently.

---

## Technologies Used

- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **SQLite**
- **HTML**
- **CSS**
- **Joblib**
- **Werkzeug**

---

## Project Structure

```text
Ecommerce-Fraud-Detection-System
│
├── app.py
├── train_model.py
├── init_db.py
├── requirements.txt
│
├── model
│   └── fraud_pipeline.joblib
│
├── dataset
│   └── ecommerce_fraud.csv
│
├── templates
│   ├── base.html
│   ├── dashboard.html
│   ├── login.html
│   ├── predict.html
│   ├── upload.html
│   ├── history.html
│   ├── suspicious.html
│   ├── simulate.html
│   ├── cases.html
│   ├── notifications.html
│   ├── risk_profiles.html
│   └── case_timeline.html
│
├── static
│   └── style.css
│
├── screenshots
│   ├── screenshot1.png
│   ├── screenshot2.png
│   ├── screenshot3.png
│
└── README.md
