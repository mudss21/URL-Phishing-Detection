"""
URL-Based Phishing Detection using Machine Learning

This script trains and evaluates multiple machine learning models
to classify URLs as phishing or legitimate.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# 1. Load Dataset
# -----------------------------
DATASET_PATH = "Dataset(2).csv"
TARGET_COLUMN = "Type"

data = pd.read_csv(DATASET_PATH)

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]


# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 3. Data Preprocessing
# -----------------------------
# Handle missing values
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 4. Model Definitions
# -----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=2,
        gamma="scale"
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Artificial Neural Network": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=80,
        random_state=42,
        early_stopping=True
    )
}


# -----------------------------
# 5. Training & Evaluation
# -----------------------------
for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 60)
