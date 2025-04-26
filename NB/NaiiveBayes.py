import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import ComplementNB 
import matplotlib.pyplot as plt

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data, target_column='fraud_bool'):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=[np.number]).columns

    # Create preprocessing steps (no polynomial features)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # Apply polynomial features transformation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_processed = preprocessor.fit_transform(X)
    X_processed = poly.fit_transform(X_processed)
    return X_processed, y

# Step 3: Split the data
def split_data(X, y, test_size=0.1, val_size=0.1, random_state=42):
    
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Adjust validation size relative to train+val
    val_ratio = val_size / (1 - test_size)

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    # Rebalance validation set to 50/50 fraud and non-fraud
    X_val = pd.DataFrame(X_val, index=y_val.index)  # Align indices for manipulation
    y_val = pd.Series(y_val, index=X_val.index)

    # Separate fraud and non-fraud cases
    val_fraud = X_val[y_val == 1]
    val_non_fraud = X_val[y_val == 0]

    num_fraud = len(val_fraud)
    val_non_fraud_downsampled = val_non_fraud.sample(n=num_fraud, random_state=random_state)

    # Combine fraud and non-fraud cases
    X_val_balanced = pd.concat([val_fraud, val_non_fraud_downsampled])
    y_val_balanced = pd.concat([y_val[y_val == 1], y_val[y_val == 0].sample(n=num_fraud, random_state=random_state)])

    return X_train, X_val_balanced.to_numpy(), X_test, y_train, y_val_balanced.to_numpy(), y_test

# Step 4: Apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# Step 5: Train a Naive Bayes model
def train_model(X_train, y_train):
    model = ComplementNB(
        alpha = 1
    )  
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate with a threshold
def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_thresholded = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_thresholded)
    conf_matrix = confusion_matrix(y_test, y_pred_thresholded)
    class_report = classification_report(y_test, y_pred_thresholded)

    print(f"Threshold: {threshold}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

# Step 7: Tune the threshold
def tune_threshold(model, X_val, y_val, threshold_range=np.arange(0.01, 0.99, 0.001)):
    y_prob = model.predict_proba(X_val)[:, 1]
    best_f1 = 0
    best_threshold = 0.5

    for threshold in threshold_range:
        y_pred_thresholded = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred_thresholded)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", color="red")
    plt.axvline(x=best_threshold, color="green", linestyle="--", label=f"Best Threshold: {best_threshold:.3f}")
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Best Threshold: {best_threshold} with F1-score: {best_f1}")
    return best_threshold

# Main function
def main(file_path="Base.csv", target_column='fraud_bool'):
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    model = train_model(X_train_smote, y_train_smote)

    best_threshold = tune_threshold(model, X_val, y_val)
    evaluate_model(model, X_test, y_test, threshold=best_threshold)

# Run the script
main("Base.csv")