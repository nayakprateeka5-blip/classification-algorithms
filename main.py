# ============================================================
# CLASSIFICATION ALGORITHMS PROJECT
# Breast Cancer Classification using Machine Learning
# ============================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import os

# STEP 2: CREATE OUTPUT FOLDER

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 65)
print("     BREAST CANCER CLASSIFICATION PROJECT")
print("=" * 65)

# STEP 3: LOAD DATASET

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

y = pd.Series(data.target, name="target")

print("\n1. DATASET INFORMATION")
print("-" * 40)
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Target classes:", list(data.target_names))

print("\nClass distribution:")
print(y.map({0: "malignant", 1: "benign"}).value_counts())

# Save dataset copy
df = X.copy()
df["target"] = y
df.to_csv("outputs/dataset.csv", index=False)

# STEP 4: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n2. TRAIN-TEST SPLIT")
print("-" * 40)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: DEFINE CLASSIFICATION ALGORITHMS

algorithms = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    ),
    "Support Vector Machine": Pipeline(
        [("scaler", StandardScaler()), ("classifier", SVC())]
    ),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# STEP 6: TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

print("\n3. TRAINING CLASSIFICATION MODELS")
print("-" * 40)

for name, model in algorithms.items():

    print(f"\nTraining: {name}")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate metrics for the benign class (label 1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Store results
    results.append(
        {
            "Algorithm": name,
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
        }
    )

    trained_models[name] = model

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# STEP 7: COMPARE ALL MODELS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1 Score", ascending=False).reset_index(
    drop=True
)

print("\n" + "=" * 65)
print("4. CLASSIFICATION ALGORITHM COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv("outputs/algorithm_comparison.csv", index=False)

print("\nComparison saved to outputs/algorithm_comparison.csv")

# STEP 8: SELECT BEST MODEL BY F1 SCORE

best_name = results_df.iloc[0]["Algorithm"]
best_model = trained_models[best_name]

print("\n5. BEST MODEL")
print("-" * 40)
print("Best algorithm by F1 Score:", best_name)
print("Best F1 Score:", results_df.iloc[0]["F1 Score"])
print("Best accuracy:", str(results_df.iloc[0]["Accuracy (%)"]) + "%")

# Save selected model
joblib.dump(best_model, "models/best_classification_model.joblib")

print("Best model saved to models/best_classification_model.joblib")

# STEP 9: DETAILED REPORT FOR BEST MODEL

best_predictions = best_model.predict(X_test)

print("\n6. CLASSIFICATION REPORT FOR BEST MODEL")
print("-" * 40)

print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        zero_division=0,
    )
)

# STEP 10: CONFUSION MATRIX

cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["malignant", "benign"]
)

disp.plot()
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# STEP 11: ACCURACY COMPARISON GRAPH

plt.figure(figsize=(11, 6))

plt.bar(results_df["Algorithm"], results_df["Accuracy (%)"])

plt.title("Classification Algorithms - Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("outputs/accuracy_comparison.png", dpi=300)
plt.show()

# STEP 12: F1 SCORE COMPARISON GRAPH

plt.figure(figsize=(11, 6))

plt.bar(results_df["Algorithm"], results_df["F1 Score"])

plt.title("Classification Algorithms - F1 Score Comparison")
plt.xlabel("Algorithm")
plt.ylabel("F1 Score")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("outputs/f1_score_comparison.png", dpi=300)
plt.show()

# STEP 13: SAMPLE PREDICTION

sample = X_test.iloc[[0]]
actual_class = y_test.iloc[0]

predicted_class = best_model.predict(sample)[0]

print("\n7. SAMPLE PREDICTION")
print("-" * 40)
print("Predicted class:", data.target_names[predicted_class])
print("Actual class:", data.target_names[actual_class])

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(sample)[0]

    print("\nPredicted class probabilities:")
    for index, class_name in enumerate(data.target_names):
        print(f"{class_name}: {probabilities[index] * 100:.2f}%")

# STEP 14: SAVE TEST PREDICTIONS

prediction_df = pd.DataFrame(
    {
        "Actual": [data.target_names[value] for value in y_test],
        "Predicted": [data.target_names[value] for value in best_predictions],
    }
)

prediction_df.to_csv("outputs/test_predictions.csv", index=False)

# STEP 15: FINISH

print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")
print("1. outputs/dataset.csv")
print("2. outputs/algorithm_comparison.csv")
print("3. outputs/test_predictions.csv")
print("4. outputs/accuracy_comparison.png")
print("5. outputs/f1_score_comparison.png")
print("6. outputs/confusion_matrix.png")
print("7. models/best_classification_model.joblib")

print("\nNote: This is an educational ML project,")
print("not a medical diagnostic tool.")
