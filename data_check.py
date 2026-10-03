import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

data = pd.read_csv("data/advertising.csv")

print("Dataset loaded successfully!")


# ==================================================
# 2. HANDLE TIMESTAMP
# ==================================================

data["Timestamp"] = pd.to_datetime(data["Timestamp"])

data["Hour"] = data["Timestamp"].dt.hour
data["Day"] = data["Timestamp"].dt.day
data["Month"] = data["Timestamp"].dt.month
data["DayOfWeek"] = data["Timestamp"].dt.dayofweek


# ==================================================
# 3. SEPARATE FEATURES AND TARGET
# ==================================================

X = data.drop("Clicked on Ad", axis=1)
y = data["Clicked on Ad"]


# ==================================================
# 4. REMOVE UNUSED COLUMNS
# ==================================================

X = X.drop(["Timestamp", "Ad Topic Line"], axis=1)


# ==================================================
# 5. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# 6. DEFINE CATEGORICAL COLUMNS
# ==================================================

categorical_columns = ["City", "Country"]


# ==================================================
# 7. CREATE PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==================================================
# 8. CREATE MACHINE LEARNING PIPELINE
# ==================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ==================================================
# 9. TRAIN THE MODEL
# ==================================================

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ==================================================
# 10. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==================================================
# 11. MODEL EVALUATION
# ==================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)


print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nROC-AUC   :", roc_auc)


# ==================================================
# 12. CONFUSION MATRIX VISUALIZATION
# ==================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Ad Click Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["Not Clicked", "Clicked"])
plt.yticks([0, 1], ["Not Clicked", "Clicked"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# 13. ROC CURVE
# ==================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Ad Click Prediction")

plt.legend()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# 14. SAVE TRAINED MODEL
# ==================================================

joblib.dump(model, "ctr_model.pkl")

print("\nModel saved successfully!")


# ==================================================
# 15. PREDICTION ON UNSEEN TEST SAMPLES
# ==================================================

# Select 15 samples from the test dataset
sample_X = X_test.iloc[15:30]
sample_y = y_test.iloc[15:30]

# Make predictions
sample_predictions = model.predict(sample_X)

# Get prediction probabilities
sample_probabilities = model.predict_proba(sample_X)[:, 1]


print("\n================================")
print("15 SAMPLE PREDICTIONS")
print("================================")

print("\nSample | Actual | Predicted | Click Probability")
print("------------------------------------------------")

for i in range(15):

    actual = sample_y.iloc[i]
    predicted = sample_predictions[i]
    probability = sample_probabilities[i]

    print(
        f"{i + 1:6} | "
        f"{actual:6} | "
        f"{predicted:9} | "
        f"{probability:.4f}"
    )


# Calculate accuracy for the 15 samples
sample_accuracy = accuracy_score(
    sample_y,
    sample_predictions
)

print("\n15-Sample Accuracy:", sample_accuracy)
print("15-Sample Accuracy (%):", sample_accuracy * 100)

# Select 30 unseen test samples
sample_X = X_test.iloc[:30]
sample_y = y_test.iloc[:30]

# Predictions
sample_predictions = model.predict(sample_X)

# Probabilities
sample_probabilities = model.predict_proba(sample_X)[:, 1]

# Calculate accuracy
sample_accuracy = accuracy_score(sample_y, sample_predictions)

print("\n30-Sample Accuracy:", sample_accuracy)
print("30-Sample Accuracy (%):", sample_accuracy * 100)

# ==================================================
# 16. FINAL MODEL SUMMARY
# ==================================================

print("\n================================")
print("FINAL MODEL PERFORMANCE")
print("================================")

print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)
print("ROC-AUC   :", roc_auc)

print("================================")