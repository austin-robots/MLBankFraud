import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, make_scorer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data, target_column='fraud_bool', poly_degree=2):
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

# Step 4: Train XGBoost Model with HalvingGridSearchCV
def train_xgboost_with_halving(X_train, y_train):
    # smote = SMOTE(random_state=42)
    # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Define XGBoost model
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [400, 500, 600],  # Number of trees
        'learning_rate': [0.1, 0.05, 1],  # Step size
        'max_depth': [8, 10, 6],  # Maximum depth of trees
        'subsample': [0.9, 0.8, 1]  # Fraction of samples per tree
    }

    # Use HalvingGridSearchCV for hyperparameter tuning
    halving_search = HalvingGridSearchCV(
        xgb,
        param_grid,
        scoring='f1',
        cv=3,
        factor=2,  # Halving factor: reduce candidates by half each iteration
        verbose=3,
        n_jobs=-1
    )

    # Fit the model
    halving_search.fit(X_train, y_train)

    # Output best parameters
    print("\nBest Parameters:", halving_search.best_params_)
    print("Best Cross-Validation F1 Score:", halving_search.best_score_)

    # Extract cv_results_ for visualization
    results_df = pd.DataFrame(halving_search.cv_results_)

    results = halving_search.cv_results_

    # Check what metric was evaluated
    print("Scorer Used:", halving_search.scorer_)

    # Output the mean test score (should match macro recall if properly configured)
    print("Mean Test Scores (from cv_results_):")
    print(results['mean_test_score'])

    # Check if the scoring metric matches the output logs
    print("GridSearch Results:")
    for i, params in enumerate(results['params']):
        print(f"Params: {params}, Mean Test Score: {results['mean_test_score'][i]}")

    # Plot the effect of n_estimators on accuracy for different max_depth values
    plt.figure(figsize=(14, 6))

    # First Plot: n_estimators vs mean_test_score for different max_depth values
    plt.subplot(1, 2, 1)
    for depth in results_df['param_max_depth'].unique():
        subset = results_df[results_df['param_max_depth'] == depth]
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', label=f'Max Depth: {depth}')

    plt.title('Effect of n_estimators on Mean Test Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Test Score')
    plt.legend()

    # Second Plot: Overall trend with threshold line (example threshold at n_estimators = 100)
    plt.subplot(1, 2, 2)
    plt.plot(results_df['param_n_estimators'], results_df['mean_test_score'], marker='o', label='Mean Test Score')

    # Add a vertical line at n_estimators = 100 (example threshold)
    plt.axvline(x=100, color='r', linestyle='--', label='Threshold Estimator')

    plt.title('Mean Test Score with Threshold Line')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Test Score')

    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

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
def main(file_path="Base.csv", target_column='fraud_bool', poly_degree=2):
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column, poly_degree)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model with HalvingGridSearchCV
    model = train_xgboost_with_halving(X_train, y_train)

    # Evaluate on the test set
    evaluate_model(model, X_test, y_test)

# Run the main function
main("Base.csv")