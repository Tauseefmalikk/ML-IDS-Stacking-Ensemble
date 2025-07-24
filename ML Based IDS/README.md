# 🛡️ ML-Based Intrusion Detection System (IDS) using Stacking Ensemble

A Machine Learning-based Intrusion Detection System developed using a **Stacking Ensemble Classifier** trained on the CICIDS 2017 dataset. The system includes an interactive **Streamlit dashboard** to test the model with custom input and visualize results, achieving up to **99% accuracy** in identifying various network attacks.

---

## 📌 Project Overview

This project aims to detect cyber threats and classify malicious network activity using supervised machine learning techniques. It utilizes a **stacking ensemble model** to improve predictive performance by combining multiple classifiers.

---

## 📂 Dataset

- **Name:** CICIDS 2017  
- **Source:** Canadian Institute for Cybersecurity  
- **Type:** Labeled network traffic with 80+ features  
- **Link:** [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## 🧠 Model Architecture

### 🔹 Base Models
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Logistic Regression
- CatBoostClassifier

### 🔹 Meta-Learner
- Random Forest

This ensemble reduces overfitting and increases generalization across unseen attack types.

---

## 🖥️ Streamlit Dashboard Features

- Upload test data (CSV)
- Predict class label (Normal / Attack)
- Visualize:
  - Prediction summary
- Input custom data manually
- Real-time feedback on model output

---

## 📈 Performance Metrics

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 99.0%   |
| Precision | 98.8%   |
| Recall    | 98.6%   |
| F1 Score  | 98.7%   |



---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tauseefmalikk/ML-IDS-Stacking-Ensemble.git
cd ML-IDS-Stacking-Ensemble

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
