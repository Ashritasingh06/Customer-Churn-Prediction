# 📊 AI-Driven Customer Churn Prediction System
## 🔍 Project Overview
Customer churn is a critical problem for businesses. This project predicts whether a customer is likely to leave (churn) using Machine Learning and provides an interactive dashboard for real-time predictions.

The project demonstrates a complete end-to-end Data Science workflow:
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Model deployment using Streamlit
---
## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Model:** Logistic Regression  
- **Dashboard:** Streamlit  
- **Tools:** Jupyter Notebook, GitHub  
---
## 📂 Project Structure
Customer-Churn-Prediction/
│
├── dashboard/
│ └── app.py # Streamlit dashboard application
│
├── data/
│ └── customer_data.csv # Dataset used for training
│
├── models/
│ └── churn_model.pkl # Trained ML model
│
├── notebooks/
│ └── 02_data_cleaning.ipynb # Data cleaning & EDA
│
├── src/
│ └── train_model.py # Model training script
│
├── requirements.txt # Project dependencies
└── README.md # Project documentation
---
## 🚀 How to Run the Project Locally
### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the Streamlit dashboard
python -m streamlit run dashboard/app.py
3️⃣ Open in browser
http://localhost:8501
🎯 Features
Interactive dashboard for churn prediction
User input-based real-time predictions
Clear churn result visualization (Stay / Churn)
Simple and clean UI suitable for business users
📈 Machine Learning Details
Algorithm Used: Logistic Regression
Target Variable: Churn (Yes / No)
Preprocessing: Label Encoding, Train-Test Split
Model Storage: joblib (.pkl file)
💡 Use Case
Helps businesses identify high-risk customers
Enables proactive customer retention strategies
Useful for telecom, banking, SaaS, and subscription-based services
👤 Author
Ashrita Singh
Data Analyst | Machine Learning Enthusiast
⭐ If you found this project useful, feel free to star the repository!
