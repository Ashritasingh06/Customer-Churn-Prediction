import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")

model = joblib.load(MODEL_PATH)

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn using Machine Learning.")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
st.subheader("📊 Churn Distribution")
st.bar_chart(df['Churn'].value_counts())


# -----------------------------
# User input
# -----------------------------
st.subheader("🔍 Predict Customer Churn")

input_data = {}

for col in df.drop("Churn", axis=1).columns:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([input_data])

# Encode inputs
from sklearn.preprocessing import LabelEncoder
for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = LabelEncoder().fit_transform(input_df[col])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")


