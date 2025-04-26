import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data, target_column='fraud_bool', poly_degree=1):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=[np.number]).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])

    # Apply polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_processed = preprocessor.fit_transform(X)
    X_processed = poly.fit_transform(X_processed)

    return X_processed, y

# Step 3: Split the data into training and testing sets
def split_data(X, y, test_size=0.5, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 4: Train with HalvingGridSearchCV and SMOTE
def train_naive_bayes_with_halving(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Define GaussianNB and hyperparameter grid
    model = GaussianNB()
    param_grid = {
        'var_smoothing': np.logspace(-9, -6, num=50)  # Range of variance smoothing
    }

    # Use HalvingGridSearchCV for hyperparameter tuning
    halving_search = HalvingGridSearchCV(
        model, 
        param_grid, 
        scoring='f1',  # Optimize for F1-score
        cv=3,          # 3-fold cross-validation
        factor=2,      # Halve candidates at each iteration
        verbose=3, 
        random_state=42
    )

    # Fit the model
    halving_search.fit(X_train_smote, y_train_smote)

    # Best estimator and parameters
    print("\nBest Parameters:", halving_search.best_params_)
    print("Best Cross-Validation Score (F1):", halving_search.best_score_)
    return halving_search.best_estimator_

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nEvaluation Results:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

# Main function to run the workflow
def main(file_path="Base.csv", target_column='fraud_bool', poly_degree=1):
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column, poly_degree)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model with HalvingGridSearchCV
    model = train_naive_bayes_with_halving(X_train, y_train)

    # Evaluate on the test set
    evaluate_model(model, X_test, y_test)

# Run the main function
main("Base.csv")