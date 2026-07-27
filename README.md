# Submission Checklist


1. GitHub Repository Link containing

- Complete source code
- requirements.txt
- A clear README.md
- Test data used in your experiments (csv)

2. Live Streamlit App Link

- Deployed using Streamlit Community Cloud
- Must open an interactive frontend when clicked

3. Screenshot

- Upload screenshot of assignment execution on BITS Virtual Lab

4. The Github README content (details mentioned in Section 3 - Step 5) should

- also be part of the submitted PDF file.

5. App loads without errors / All required features implemented

- Screenshot link 

6. README.md updated and added in the submitted PDF

- Link to this Readme is given

# Breast Cancer Diagnostic Classification Dashboard

An interactive machine learning web application built with **Streamlit** that classifies breast tumours as **Benign** or **Malignant** using five classification algorithms. The dashboard provides live model evaluation, side-by-side model comparison, confusion matrix visualisation, and an adjustable decision threshold.

---

## Project Structure

```
AIML-2/
├── app.py                    # Streamlit dashboard (main application)
├── data.csv                  # Wisconsin Breast Cancer dataset (569 samples, 30 features)
├── test_data.csv             # Auto-generated 20% test split used by the app
├── requirements.txt          # Python dependencies (pinned versions)
└── model/
    ├── train_and_save.py     # Training pipeline — run this first
    ├── scaler.pkl            # Fitted StandardScaler artifact
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── knn_model.pkl
    ├── naive_bayes_model.pkl
    └── random_forest_model.pkl
```

---

## Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / Kaggle |
| Samples | 569 |
| Features | 30 numeric (cell nucleus measurements: mean, SE, worst) |
| Target | `diagnosis` — `M` (Malignant = 1) / `B` (Benign = 0) |
| Class balance | 357 Benign (63%) / 212 Malignant (37%) |

Dropped during preprocessing: `id` (identifier), `Unnamed: 32` (empty column).

---

## Setup & Installation

### 1. Clone / navigate to the project
```bash
cd /Users/yogeshthosare/Work/AIML-2
```

### 2. Install dependencies
> ⚠️ **Important:** Use the same Python that runs Streamlit. scikit-learn is pinned to `1.4.0` to prevent version mismatch errors when loading `.pkl` files.

```bash
pip install -r requirements.txt
```

### 3. Train and save models
```bash
/opt/homebrew/bin/python3.11 model/train_and_save.py
```

This will:
- Load and preprocess `data.csv`
- Perform a stratified 80/20 train-test split
- Fit and save all 5 models + scaler to `model/`
- Save `test_data.csv` for default app evaluation

### 4. Run the app
```bash
/opt/homebrew/bin/streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Training Pipeline (`model/train_and_save.py`)

| Step | Detail |
|------|--------|
| Load | Reads `data.csv` |
| Clean | Drops `id` and `Unnamed: 32` columns |
| Encode | Maps `diagnosis`: `M → 1`, `B → 0` |
| Split | Stratified 80/20 train-test split (`random_state=42`) |
| Scale | `StandardScaler` fitted on training set only |
| Train | Fits 5 classifiers, saves each as `.pkl` |
| Export | Saves `scaler.pkl` and `test_data.csv` |

### Models Trained

| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=5` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Naive Bayes | GaussianNB (default) |
| Random Forest | `n_estimators=100` |

---

## Dashboard Features (`app.py`)

### Sidebar Controls
- **Model selector** — choose any of the 5 trained classifiers
- **Decision threshold slider** (0.10 – 0.90, default 0.50) — adjusts the probability cutoff for classifying a tumour as Malignant
- **CSV uploader** — upload your own test file; falls back to `test_data.csv` automatically

### Main Panel
- **Data Preview** — first 5 rows of the loaded test set
- **Selected Model Metrics** — live metric cards: Accuracy, AUC, Precision, Recall, F1, MCC
- **All Models Comparison Table** — all 5 models evaluated side-by-side at the current threshold
- **Confusion Matrix** — annotated heatmap with TN / FP / FN / TP labels
- **Classification Report** — per-class (Benign / Malignant) precision, recall, F1, support

### Decision Threshold
The threshold converts raw probability scores into hard labels:
- **Lower threshold (e.g. 0.30)** → higher Recall — catches more Malignant cases (fewer missed cancers)
- **Higher threshold (e.g. 0.70)** → higher Precision — fewer false alarms

This is especially important because missing a Malignant case (false negative) has higher clinical cost than a false positive.

---

## Uploading Test Data

The app accepts any CSV with the same 30 feature columns as `data.csv`. Two modes:

| CSV contains `diagnosis` column | Behaviour |
|----------------------------------|-----------|
| ✅ Yes (numeric `0`/`1` or string `B`/`M`) | Full evaluation with metrics, confusion matrix, classification report |
| ❌ No | Inference mode — outputs `Predicted_Diagnosis` and `Malignant_Probability` columns |

A ready-to-use sample is `test_data.csv` (auto-generated by the training script).

---

## Dependency Note

`scikit-learn` is pinned to `==1.4.0` in `requirements.txt`. Always train models using the **same Python environment** that runs Streamlit. Mixing environments (e.g. Anaconda vs Homebrew) causes `InconsistentVersionWarning` and `AttributeError` when loading `.pkl` files.

```bash
# Always retrain with the same Python as Streamlit
/opt/homebrew/bin/python3.11 model/train_and_save.py
```

---

## Understanding Positive / Negative in Medical Classification

In medical diagnostic testing and machine learning classification:

### 🔴 Positive (1) = Malignant (Cancerous)
Why? In medical testing, a "Positive" result means the condition being tested for (cancer/disease) is present.

**Model Target: Class 1**

| Term | Meaning |
|------|---------|
| **True Positive (TP)** | The patient has a malignant tumour, and the model correctly flags it as malignant. |
| **False Positive (FP)** | The patient has a benign growth, but the model incorrectly flags it as malignant (a false alarm). |

### 🟢 Negative (0) = Benign (Non-Cancerous)
Why? A "Negative" medical result means the patient is free of the condition being screened for (no cancer detected).

**Model Target: Class 0**

| Term | Meaning |
|------|---------|
| **True Negative (TN)** | The growth is benign, and the model correctly labels it benign. |
| **False Negative (FN)** | The patient has a malignant tumour, but the model mistakenly flags it as benign (**the most dangerous error in healthcare**). |

> 💡 **Key Takeaway:** "Positive" in classification does not mean "good news." It simply means "the target event or disease was detected."
