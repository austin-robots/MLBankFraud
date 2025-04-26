import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Step 1: Load the dataset
def load_data(file_path):
    # Load the data from a CSV file
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data, target_column='fraud_bool', poly_degree=2):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=[np.number]).columns

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Apply polynomial features transformation
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_processed = preprocessor.fit_transform(X)
    X_processed = poly.fit_transform(X_processed)

    return X_processed, y

# Step 3: Split the data into training and testing sets, rebalancing the test set
def split_data(X, y, test_size=0.90, random_state=42):
    # Perform the initial train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Convert X_test and y_test into DataFrame/Series for easier manipulation
    X_test = pd.DataFrame(X_test, index=y_test.index)  # Ensure indices align
    y_test = pd.Series(y_test, index=X_test.index)

    # Rebalance the test set to have equal fraud and non-fraud cases
    test_fraud = X_test[y_test == 1]
    test_non_fraud = X_test[y_test == 0]

    num_fraud = len(test_fraud)
    test_non_fraud_downsampled = test_non_fraud.sample(n=num_fraud, random_state=random_state)

    # Combine the fraud and non-fraud cases
    X_test_balanced = pd.concat([test_fraud, test_non_fraud_downsampled])
    y_test_balanced = pd.concat([y_test[y_test == 1], y_test[y_test == 0].sample(n=num_fraud, random_state=random_state)])

    return X_train, X_test_balanced.to_numpy(), y_train, y_test_balanced.to_numpy()

# Step 4: Apply SMOTE only to the training data
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

# Step 5: Train a logistic regression model with Grid Search
def train_model_with_grid_search(X_train, y_train):
    logistic = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=3000, random_state=42)

    # Define hyperparameter grid for Grid Search
    param_grid = {
        'C': [3, 2.5, 2, 1.5, 1],
    }

    # Perform grid search directly on the model
    grid_search = GridSearchCV(
        estimator=logistic,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=3,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("ROC-AUC Score:", roc_auc)

# Main function to run the entire workflow
def main(file_path="Base.csv", target_column='fraud_bool', poly_degree=2):
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column, poly_degree)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply SMOTE only to the training set
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Train the model with grid search
    model = train_model_with_grid_search(X_train_smote, y_train_smote)
    evaluate_model(model, X_test, y_test)

# Run the main function with Base.csv as the data file
main("Base.csv")
