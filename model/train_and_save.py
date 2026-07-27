import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('data.csv')

# Drop non-feature / empty columns
cols_to_drop = []
if 'id' in df.columns:
    cols_to_drop.append('id')
if 'Unnamed: 32' in df.columns:
    cols_to_drop.append('Unnamed: 32')

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# 2. Encode Target Variable (Malignant = 1, Benign = 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. Separate Features and Target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 4. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test dataset locally for app default fallback
test_df = X_test.copy()
test_df['diagnosis'] = y_test
test_df.to_csv("test_data.csv", index=False)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save Scaler
with open(os.path.join("model", "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# 6. Model Definitions
models = {
    "logistic_regression_model.pkl": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree_model.pkl": DecisionTreeClassifier(random_state=42, max_depth=5),
    "knn_model.pkl": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes_model.pkl": GaussianNB(),
    "random_forest_model.pkl": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 7. Fit & Save Each Model
print("Training models...")
for filename, model in models.items():
    model.fit(X_train_scaled, y_train)
    with open(os.path.join("model", filename), "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

print("\nAll models and scale artifacts successfully saved!")