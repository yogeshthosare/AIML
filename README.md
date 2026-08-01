# Submission Checklist

1. GitHub Repository Link containing: https://github.com/yogeshthosare/AIML

- Complete source code - Done
- requirements.txt - Done
- A clear README.md - Done
- Test data used in your experiments (csv) - test_data.csv Done
  Application directly uses default data test_data.csv taken from Train-Test split when webpage is opened, use upload test data option to upload test data. 

>link: https://github.com/yogeshthosare/AIML

2. Live Streamlit App Link


- Deployed using Streamlit Community Cloud - Done
- Must open an interactive frontend when clicked - Done

>link : https://yogeshthosare-aiml-app-y6g6jc.streamlit.app/

3. Screenshot

- Upload screenshot of assignment execution on BITS Virtual Lab
- App loads without errors / All required features implemented

>link 1: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_01.png

>link 2: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_02.png

>link 3: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_03.png

>link 4: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_04.png

>link 5: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_05.png

>link 6: https://github.com/yogeshthosare/AIML/blob/main/LogisticRegression.png

>link 7: https://github.com/yogeshthosare/AIML/blob/main/DecisionTree.png

>link 8: https://github.com/yogeshthosare/AIML/blob/main/KNN.png

>link 9: https://github.com/yogeshthosare/AIML/blob/main/NaiveBayes.png

>link 10: https://github.com/yogeshthosare/AIML/blob/main/RandomForest.png

4. The Github README content (details mentioned in Section 3 - Step 5)

- should also be part of the submitted PDF file.

>link: https://github.com/yogeshthosare/AIML/blob/main/README.md

# Assignment Details

## Step 1: Dataset Choice - Done

>link: https://www.kaggle.com/code/anandhuh/breast-cancer-prediction-accuracy-98-24/input

## Step 2: Machine Learning Classification models and Evaluation metrics - Done

>link: https://github.com/yogeshthosare/AIML/blob/main/All_Models_Performance_Comparison

## Step 3: Prepare Your GitHub Repository - Done

>link: https://github.com/yogeshthosare/AIML

## Step 4: Create requirements.txt - Done

>link: https://github.com/yogeshthosare/AIML/blob/main/requirements.txt

## Step 5: README.md with the following structure

### a. Problem statement

#### Breast Cancer Diagnostic Classification Dashboard

An interactive machine learning web application built with **Streamlit** that classifies breast tumours as **Benign** or **Malignant** using five classification algorithms. The dashboard provides live model evaluation, side-by-side model comparison, confusion matrix visualisation, and an adjustable decision threshold is also supported.

---

### b. Dataset description

**Wisconsin Breast Cancer Diagnostic Dataset**

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / Kaggle |
| Samples | 569 |
| Features | 30 numeric (cell nucleus measurements: mean, SE, worst) |
| Target | `diagnosis` — `M` (Malignant = 1) / `B` (Benign = 0) |
| Class balance | 357 Benign (63%) / 212 Malignant (37%) |

Dropped during preprocessing: `id` (identifier), `Unnamed: 32` (empty column).

### c. Github Repository Link [for maintaining the github repo with all required files]

>link: https://github.com/yogeshthosare/AIML

```
ML-ASSIGNMENT-SECONDSEM
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

### d. Models Used: [ 5 marks - 1 marks for all the metrics for each model ] 

- Comparison Table with the evaluation metrics calculated for all the models

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree** | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| **K-Nearest Neighbors (KNN)** | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| **Naive Bayes** | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| **Random Forest** | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

>link: https://github.com/yogeshthosare/AIML/blob/main/All_Models_Performance_Comparison

- Observations on the performance of each model on the chosen dataset.

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly with high overall accuracy (96.49%), tied for the highest recall (92.86%), and achieved the highest AUC score (0.996), making it a top-tier linear baseline. |
| **Decision Tree** | Lowest overall performing model with the lowest recall (83.33%) and AUC score (0.945), indicating high false negatives and potential overfitting. |
| **kNN** | Delivered strong performance with high accuracy (95.61%) and high precision (97.44%), effective at minimizing false positives. |
| **Naive Bayes** | Achieved good AUC score (0.989) indicating strong class separation, but had lower precision (92.31%) and recall (85.71%) relative to top models. |
| **Random Forest (Ensemble)** | Top-performing model across all key metrics, achieving the highest accuracy (97.37%) and perfect precision (1.00) with zero false positives. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** yields the best overall accuracy (97.37%), highest F1 score (0.963), and perfect precision without compromising on recall (92.86%). |

## Step 6: Deploy on Streamlit Community Cloud

>link : https://yogeshthosare-aiml-app-y6g6jc.streamlit.app/

### a. Dataset upload option (CSV) [As streamlit free tier has limited capacity, upload only test data] [ 1 mark ]

2 test datasets are given in the github repository, those can be uploaded from UI.

>link 1: https://github.com/yogeshthosare/AIML/blob/main/test_data.csv
>link 2: https://github.com/yogeshthosare/AIML/blob/main/test_data_20.csv

### b. Model selection dropdown (if multiple models) [ 1 mark ]

>link: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_05.png

### c. Display of evaluation metrics [ 1 mark ]

>link: https://github.com/yogeshthosare/AIML/blob/main/All_Models_Performance_Comparison

### d. Confusion matrix or classification report [ 1 mark ]

>link: https://github.com/yogeshthosare/AIML/blob/main/Streamlit_Application_BITS_Virtual_Lab_03.png

- The results of different models on your “test data” should be visible on the streamlit app.

>link 1: https://github.com/yogeshthosare/AIML/blob/main/LogisticRegression.png

>link 2: https://github.com/yogeshthosare/AIML/blob/main/DecisionTree.png

>link 3: https://github.com/yogeshthosare/AIML/blob/main/KNN.png

>link 4: https://github.com/yogeshthosare/AIML/blob/main/NaiveBayes.png

>link 5: https://github.com/yogeshthosare/AIML/blob/main/RandomForest.png

## Setup & Installation

### 1. Clone / navigate to the project
```bash
cd AIML
```

### 2. Install dependencies
> ⚠️ **Important:** Use the same Python that runs Streamlit. scikit-learn is pinned to `1.4.0` to prevent version mismatch errors when loading `.pkl` files.

```bash
pip install -r requirements.txt
```

### 3. Train and save models
```bash
python3.11 model/train_and_save.py
```

This will:
- Load and preprocess `data.csv`
- Perform a stratified 80/20 train-test split
- Fit and save all 5 models + scaler to `model/`
- Save `test_data.csv` for default app evaluation

### 4. Run the app
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

To deploy on Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign in using GitHub account
3. Click “New App”
4. Select your repository
5. Choose branch (usually main)
6. Select app.py
7. Click Deploy

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

### How the App Behaves Based on Your CSV Input

In both modes below, the app runs **100% inference** using pre-trained models. No model training occurs inside the app.

---

#### 1. If your CSV includes the `diagnosis` column (Test / Benchmark Mode)
* **What happens:** The app runs predictions on the data, then compares those predictions directly against the true labels provided in the `diagnosis` column.
* **What you see in the UI:** 
  * Full evaluation metrics cards (**Accuracy**, **AUC**, **Precision**, **Recall**, **F1 Score**, **MCC**).
  * Interactive **Confusion Matrix** heatmap.
  * **Classification Report** table.
  * **All-Models Performance Comparison Table**.

---

#### 2. If your CSV does NOT include the `diagnosis` column (Pure Inference Mode)
* **What happens:** The app runs predictions on unseen patient data. Because there are no ground-truth labels to evaluate performance against, metric evaluation is skipped.
* **What you see in the UI:**
  * An informational message indicating inference mode.
  * A preview table showing the input data appended with two new output columns:
    * **`Predicted_Diagnosis`** (e.g., *Malignant* or *Benign*)
    * **`Malignant_Probability`** (e.g., *0.85* or *85%*)

A ready-to-use sample is `test_data.csv` (auto-generated by the training script).

---

## Dependency Note

`scikit-learn` is pinned to `==1.4.0` in `requirements.txt`. Always train models using the **same Python environment** that runs Streamlit. Mixing environments (e.g. Anaconda vs Homebrew) causes `InconsistentVersionWarning` and `AttributeError` when loading `.pkl` files.

```bash
# Always retrain with the same Python as Streamlit
python3.11 model/train_and_save.py
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
