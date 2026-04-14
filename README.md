# 📊 Customer Churn Prediction System

End-to-end ML project predicting customer churn with a live Streamlit web app.

🔗 Live Demo: https://customer-churn-prediction-yzd44im8swehxkofhswxyx.streamlit.app/

## Problem Statement
Customer churn costs businesses millions. This project builds a predictive model 
to identify at-risk customers before they leave.

## Tech Stack
- Language: Python
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- ML Model: Scikit-learn (Logistic Regression)
- Deployment: Streamlit
- Model Storage: joblib

## Workflow
1. Data Cleaning — missing values, duplicates
2. EDA — churn pattern analysis with visualizations
3. Feature Engineering — Label Encoding for categorical variables
4. Model Training — Logistic Regression with Train-Test Split
5. Evaluation — Confusion Matrix, Classification Report, ROC-AUC
6. Deployment — Live Streamlit app for real-time predictions

## Results
- Accuracy: ~80%
- Dataset: ~7,000 customer records

## Run Locally
pip install -r requirements.txt
python -m streamlit run dashboard/app.py

## Author
Ashrita Singh | https://www.linkedin.com/in/ashrita-singh
