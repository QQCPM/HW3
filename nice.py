"""
This program loads the CAD.csv dataset and develops supervised ML models
to predict Coronary Artery Disease (CAD) diagnosis.

Two model architectures are implemented:
1. Logistic Regression (Linear Model)
2. Random Forest (Decision Tree Ensemble)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Load the CAD dataset and preprocess it for ML models.
    """
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {df.shape[1] - 1}")  # Excluding target
    
    # Display first few rows
    print(f"\nFirst 5 rows of the dataset:")
    print(df.head())
    
    
    # It contains 'Cad' or 'Normal'
    target_column = df.columns[-1]
    print(f"\nTarget column: '{target_column}'")
    print(f"Target distribution:")
    print(df[target_column].value_counts())
    
    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column (Cath)
    
    # Encode target variable: 'Cad' -> 1, 'Normal' -> 0
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"\nTarget encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X, y_encoded, label_encoder

def partition_data(X, y, test_size=0.2, random_state=42):
    """
    Partition the dataset into training and test sets.
    """
    print("\n" + "=" * 60)
    print("PARTITIONING DATA")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Test set size: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = 'Cad' if u == 0 else 'Normal'
        print(f"  {label}: {c} ({c/len(y_train)*100:.1f}%)")
    
    print(f"\nTest set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        label = 'Cad' if u == 0 else 'Normal'
        print(f"  {label}: {c} ({c/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    Fit on training data only to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Model Architecture 1: Logistic Regression (Linear Model)
    
    Parameters chosen:
    - max_iter=1000: Ensure convergence
    - C=1.0: Default regularization strength
    - solver='lbfgs': Good for small datasets
    - class_weight='balanced': Handle any class imbalance
    """
    print("\n" + "=" * 60)
    print("MODEL 1: LOGISTIC REGRESSION (Linear Model)")
    print("=" * 60)
    
    print("\nModel Parameters:")
    print("  - max_iter: 1000 (ensure convergence)")
    print("  - C: 1.0 (regularization strength)")
    print("  - solver: 'lbfgs' (efficient for small datasets)")
    print("  - class_weight: 'balanced' (handle class imbalance)")
    
    # Initialize and train the model
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    
    print("\nTraining the model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy on TEST partition only
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n*** TEST SET ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%) ***")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Cad', 'Normal']))
    
    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                Cad    Normal")
    print(f"Actual Cad      {cm[0][0]:<6} {cm[0][1]}")
    print(f"       Normal   {cm[1][0]:<6} {cm[1][1]}")
    
    # Check for underfitting by comparing train and test accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    if train_accuracy < 0.6:
        print("WARNING: Model may be underfitting (low training accuracy)")
    elif abs(train_accuracy - accuracy) > 0.15:
        print("NOTE: Significant gap between train/test accuracy - possible overfitting")
    else:
        print("Model appears to be well-fitted (no significant under/overfitting)")
    
    return model, accuracy

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Model Architecture 2: Random Forest (Decision Tree Ensemble)
    
    Parameters chosen:
    - n_estimators=100: Number of trees in the forest
    - max_depth=10: Prevent overfitting
    - min_samples_split=5: Minimum samples to split a node
    - min_samples_leaf=2: Minimum samples in leaf nodes
    - class_weight='balanced': Handle class imbalance
    """
    print("\n" + "=" * 60)
    print("MODEL 2: RANDOM FOREST (Decision Tree Ensemble)")
    print("=" * 60)
    
    print("\nModel Parameters:")
    print("  - n_estimators: 100 (number of trees)")
    print("  - max_depth: 10 (prevent overfitting)")
    print("  - min_samples_split: 5 (minimum samples to split)")
    print("  - min_samples_leaf: 2 (minimum samples in leaf)")
    print("  - class_weight: 'balanced' (handle class imbalance)")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("\nTraining the model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy on TEST partition only
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n*** TEST SET ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%) ***")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Cad', 'Normal']))
    
    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                Cad    Normal")
    print(f"Actual Cad      {cm[0][0]:<6} {cm[0][1]}")
    print(f"       Normal   {cm[1][0]:<6} {cm[1][1]}")
    
    # Check for underfitting
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    if train_accuracy < 0.6:
        print("WARNING: Model may be underfitting (low training accuracy)")
    elif abs(train_accuracy - accuracy) > 0.15:
        print("NOTE: Significant gap between train/test accuracy - possible overfitting")
    else:
        print("Model appears to be well-fitted (no significant under/overfitting)")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, accuracy

def main():
    """
    Main function to run the CAD diagnosis ML pipeline.
    """
    print("\n" + "=" * 60)
    print("CAD DIAGNOSIS - MACHINE LEARNING APPROACH")
    print("ECE4524 Fall 2025 - Final Exam Part 2")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    filepath = "CAD.csv"
    X, y, label_encoder = load_and_preprocess_data(filepath)
    
    # Step 2: Partition data into training and test sets
    X_train, X_test, y_train, y_test = partition_data(X, y, test_size=0.2)
    
    # Step 3: Scale features (important for Logistic Regression)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 4: Train and evaluate Model 1 - Logistic Regression
    lr_model, lr_accuracy = train_logistic_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Step 5: Train and evaluate Model 2 - Random Forest
    # Note: Random Forest doesn't require scaled features, but using them doesn't hurt
    rf_model, rf_accuracy = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    # Step 6: Summary comparison
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - MODEL COMPARISON")
    print("=" * 60)
    print(f"\nModel 1 (Logistic Regression) Test Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
    print(f"Model 2 (Random Forest) Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    if rf_accuracy > lr_accuracy:
        print(f"\nRandom Forest outperforms Logistic Regression by {(rf_accuracy-lr_accuracy)*100:.2f}%")
    elif lr_accuracy > rf_accuracy:
        print(f"\nLogistic Regression outperforms Random Forest by {(lr_accuracy-rf_accuracy)*100:.2f}%")
    else:
        print("\nBoth models have equal accuracy")
    
    print("\n" + "=" * 60)
    print("PROGRAM EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
