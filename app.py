import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)

# Page Configuration
st.set_page_config(page_title="Breast Cancer Classification App", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Diagnostic Dashboard")
st.markdown("Upload test patient metrics, adjust decision thresholds, evaluate machine learning models, and compare diagnostic performance.")

# Sidebar Configuration
st.sidebar.header("⚙️ Application Configuration")

# Model Selection Dropdown
model_options = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors (KNN)": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

# Decision Threshold Slider for Precision / Recall Control
st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Decision Threshold")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Lower threshold increases Recall/Sensitivity (catches more true Malignant cases). Higher threshold increases Precision."
)

# Data Upload
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (e.g., test_data.csv)", type=["csv"])

# Helper Functions to Load Artifacts
@st.cache_resource
def load_scaler():
    scaler_path = os.path.join("model", "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join("model", model_filename)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

scaler = load_scaler()
model_file = model_options[selected_model_name]
model = load_model(model_file)

# Main Application Logic
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success("Test dataset successfully uploaded!")
else:
    # Fallback to local test_data.csv if uploaded file is absent
    if os.path.exists("test_data.csv"):
        df_test = pd.read_csv("test_data.csv")
        st.info("Using default `test_data.csv` for evaluation, use `upload test data` option on the left to upload your test data")
    else:
        st.warning("Please upload a test dataset CSV file to proceed.")
        df_test = None

if df_test is not None and model is not None and scaler is not None:
    st.markdown("---")
    st.subheader("🔍 Data Preview, target column `diagnosis`")
    st.dataframe(df_test.head(5), use_container_width=True)

    # Clean non-feature columns if present
    X_test = df_test.copy()
    for c in ['id', 'Unnamed: 32']:
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])

    # Extract ground truth 'diagnosis' if present
    if 'diagnosis' in X_test.columns:
        y_true = X_test['diagnosis']
        # Handle string encoding if user uploads raw 'M'/'B' CSV
        if y_true.dtype == object or isinstance(y_true.iloc[0], str):
            y_true = y_true.map({'M': 1, 'B': 0})
        X_test = X_test.drop(columns=['diagnosis'])
    else:
        y_true = None

    # Scale Features
    X_test_scaled = scaler.transform(X_test)

    # Model Predictions with Threshold Control
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)

    # Performance Evaluation
    if y_true is not None:
        st.markdown("---")
        st.subheader(f"📈 Selected Model Metrics: {selected_model_name} (Threshold: {threshold:.2f})")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
        mcc = matthews_corrcoef(y_true, y_pred)

        # Display Key Metrics as Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        # Plot Confusion Matrix with Explicit Annotations
        with col_left:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            # Unpack values from 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()

            # Create explicit cell annotations with metric labels and actual counts
            labels_tl = np.array([
                [f"True Negatives\n(TNs)\n{tn:,}", f"False Positives\n(FPs)\n{fp:,}"],
                [f"False Negatives\n(FNs)\n{fn:,}", f"True Positives\n(TPs)\n{tp:,}"]
            ])

            fig, ax = plt.subplots(figsize=(5, 4.5))
            sns.heatmap(
                cm,
                annot=labels_tl,
                fmt="",
                cmap="Reds",
                cbar=False,
                ax=ax,
                annot_kws={"size": 10, "weight": "bold"},
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant']
            )
            plt.ylabel('Actual Diagnosis', fontsize=11, fontweight='bold')
            plt.xlabel('Predicted Diagnosis', fontsize=11, fontweight='bold')
            plt.title(f'Confusion Matrix - {selected_model_name}', fontsize=12, fontweight='bold', pad=10)
            st.pyplot(fig)

        # Display Classification Report Table
        with col_right:
            st.subheader("📋 Classification Report")
            report_dict = classification_report(
                y_true,
                y_pred,
                target_names=['Benign', 'Malignant'],
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()

            # Filter strictly the two class rows
            class_report_df = report_df.loc[['Benign', 'Malignant']].copy()

            st.dataframe(
                class_report_df.style.format({
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:,.0f}"
                }),
                use_container_width=True
            )


        st.markdown("---")

        # Display All Models Comparison Table
        st.subheader(f"📊 All Models Performance Comparison (Threshold: {threshold:.2f})")
        comparison_results = []
        for name, filename in model_options.items():
            m_obj = load_model(filename)
            if m_obj is not None:
                yp_prob = m_obj.predict_proba(X_test_scaled)[:, 1] if hasattr(m_obj, "predict_proba") else None
                if yp_prob is not None:
                    yp = (yp_prob >= threshold).astype(int)
                else:
                    yp = m_obj.predict(X_test_scaled)

                comparison_results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_true, yp),
                    "AUC Score": roc_auc_score(y_true, yp_prob) if yp_prob is not None else np.nan,
                    "Precision": precision_score(y_true, yp, zero_division=0),
                    "Recall": recall_score(y_true, yp, zero_division=0),
                    "F1 Score": f1_score(y_true, yp, zero_division=0),
                    "MCC Score": matthews_corrcoef(y_true, yp)
                })

        comp_df = pd.DataFrame(comparison_results)
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC Score": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC Score": "{:.4f}"
            }),
            use_container_width=True
        )
    else:
        st.info("Uploaded dataset does not contain target column 'diagnosis'. Displaying generated predictions below:")
        df_results = df_test.copy()
        df_results['Predicted_Diagnosis'] = np.where(y_pred == 1, 'Malignant', 'Benign')
        if y_proba is not None:
            df_results['Malignant_Probability'] = y_proba
        st.dataframe(df_results.head(10), use_container_width=True)
else:
    if model is None or scaler is None:
        st.error("Model or Scaler pickle files were not found in the 'model/' folder. Please run `python train_and_save.py` first.")