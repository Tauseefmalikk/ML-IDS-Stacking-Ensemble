import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json

st.set_page_config(page_title="IDS Model Dashboard", layout="wide")

# -----------------------------
# Define available models
# -----------------------------
models_info = {
    "Stacking Ensemble": {
        "model_file": "StackingEnsemble.joblib",
        "metrics_file": "StackingEnsemble_metrics.json"
    },
    "Random Forest": {
        "model_file": "RandomForestClassifier.joblib",
        "metrics_file": "RandomForestClassifier_metrics.json"
    },
    "Logistic Regression": {
        "model_file": "LogisticRegression.joblib",
        "metrics_file": "LogisticRegression_metrics.json"
    },
    "Naive Bayes": {
        "model_file": "GaussianNB.joblib",
        "metrics_file": "GaussianNB_metrics.json"
    },
    "K-Nearest Neighbors": {
        "model_file": "KNeighborsClassifier.joblib",
        "metrics_file": "KNeighborsClassifier_metrics.json"
    },
    "Decision Tree": {
        "model_file": "DecisionTreeClassifier.joblib",
        "metrics_file": "DecisionTreeClassifier_metrics.json"
    }
}


# UI – Title + Model Selector

st.title("🔐 ML Based Intrusion Detection System")
model_choice = st.selectbox("🔁 Select a Model", list(models_info.keys()))
model_data = models_info[model_choice]


# Load model

model_path = os.path.join("models", model_data["model_file"])
loaded_model = joblib.load(model_path)


# Load and Display Metrics

metrics_path = os.path.join("models", model_data["metrics_file"])
try:
    with open(metrics_path) as f:
        metrics = json.load(f)
    st.markdown(f"### 📊 Performance of {model_choice}")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{metrics['training_accuracy'] * 100:.2f}%")
    col2.metric("Testing Accuracy", f"{metrics['testing_accuracy'] * 100:.2f}%")
except:
    st.warning("⚠️ Metrics not found.")


# Input Features Setup

st.markdown("## 📥 Input Features")

feature_order = [
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Max',
    ' Avg Fwd Segment Size',
    ' Subflow Fwd Bytes',
    'Total Length of Fwd Packets',
    ' Flow IAT Max',
    ' Average Packet Size',
    ' Bwd Packet Length Std',
    ' Flow Duration',
    ' Avg Bwd Segment Size',
    ' Bwd Packets/s',
    ' Packet Length Mean',
    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',
    ' Packet Length Std',
    ' Fwd IAT Max',
    ' Fwd Packet Length Std',
    ' Packet Length Variance',
    ' Total Length of Bwd Packets',
    ' Flow Packets/s'
]

class_mapping_reverse = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenEye',
    4: 'DoS Hulk',
    5: 'DoS Slowhttptest',
    6: 'DoS slowloris',
    7: 'FTP-Patator',
    8: 'Heartbleed',
    9: 'Infiltration',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack – Brute Force',
    13: 'Web Attack – Sql Injection',
    14: 'Web Attack – XSS'
}

tab1, tab2 = st.tabs(["🧪 Manual Input", "📂 CSV Upload"])

# -----------------------------
# Tab 1 – Manual Input
# -----------------------------
with tab1:
    user_input = {}
    for feature in feature_order:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

    if st.button("🔍 Predict from Input"):
        input_df = pd.DataFrame([user_input])
        try:
            prediction = loaded_model.predict(input_df)[0]
            decoded = class_mapping_reverse.get(prediction, "Unknown")
            st.success(f"Prediction: **{decoded}**")
        except Exception as e:
            st.error(f" Error: {e}")

# -----------------------------
# Tab 2 – CSV Upload
# -----------------------------
with tab2:
    uploaded_file = st.file_uploader("📄 Upload CSV file", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in feature_order):
                st.error("Uploaded file does not contain required features.")
            else:
                predictions = loaded_model.predict(df[feature_order])
                df['Predicted Class'] = [class_mapping_reverse.get(p, "Unknown") for p in predictions]
                st.success("✅ Predictions generated!")
                st.dataframe(df)
                # Download option
                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions as CSV", csv_download, file_name='predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Error processing file: {e}")
