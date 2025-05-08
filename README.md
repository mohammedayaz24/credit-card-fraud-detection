# credit-card-fraud-detection

# 🛡️ Credit Card Fraud Detection Using Machine Learning

## 📄 Project Overview
This project addresses the growing threat of credit card fraud, which poses significant financial and reputational risks to users and institutions. By leveraging advanced machine learning and anomaly detection techniques, we aim to develop a real-time fraud detection and alert system that enhances the security of digital transactions.

## 🎯 Objectives
- Analyze credit card transaction data for patterns and anomalies.
- Build machine learning models to detect fraudulent transactions with high accuracy.
- Compare and select the best model based on performance metrics.
- Simulate a real-time fraud detection alert system.
- Improve detection using ensemble learning and anomaly detection methods.

## 📌 Scope
- Supervised and unsupervised ML methods for fraud detection.
- Feature analysis includes transaction amount, location, time, and user behavior.
- Use of anonymized data due to real-world privacy constraints.
- Offline prototype for fraud detection (not linked to live systems).
- Deployment through a local dashboard (e.g., Streamlit or Flask).

## 🗃️ Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features**: 31 total — anonymized features (V1 to V28), `Amount`, `Time`, and binary `Class` label.

## 🔧 Tools & Technologies
- **Language**: Python
- **IDE/Notebook**: Google Colab / Jupyter / VS Code
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Modeling: `scikit-learn`, `xgboost`, `imbalanced-learn`, `tensorflow`/`keras`
  - Evaluation: `sklearn.metrics`
- **Optional Deployment**: `Streamlit`, `Flask`

## 🧠 Methodology
1. **Data Collection**: From Kaggle.
2. **Data Cleaning**: Handle imbalance using SMOTE, check for nulls, remove duplicates, normalize features.
3. **EDA**: Visualize data using heatmaps, histograms, etc.
4. **Feature Engineering**: Create new features from time/frequency; use PCA if needed.
5. **Model Building**: Evaluate models:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Isolation Forest
   - Autoencoders (for anomaly detection)
6. **Model Evaluation**: Use metrics like Precision, Recall, F1-Score, ROC-AUC.
7. **Interpretability**: Confusion matrix, SHAP, ROC curves.
8. **Deployment**: Local alert system using Streamlit/Flask (Prototype).

## 👥 Team Members

| Name                  | Reg. No          | Role                                      |
|-----------------------|------------------|-------------------------------------------|
| Mohammed Ayaz. A      | 510623104056     | Team Lead & Model Building                |
| Mohammed Azhan. U    | 510623104057     | Data Collection & Preprocessing           |
| Md Faazil Ammar. P    | 510623104047     | EDA & Visualization                       |
| Kashif Ulhaq. K       | 510623104040     | Feature Engineering & Dimensionality      |
| Ashfaq Ahmed. M       | 510623104009     | Model Evaluation & Validation             |
| Abrar Ul Haque. R     | 510623104004     | Report Writing & Presentation             |

## 📅 Submission Info
- **Institution**: C. Abdul Hakeem College of Engineering and Technology
- **Department**: Computer Science & Engineering
- **Submitted By**: Mohammed Ayaz. A

