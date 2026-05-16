import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess

# Hardcoded path to model
MODEL_PATH = "artifacts/Random_Forest.pkl"  # Change based on your best model

# Load and preprocess full data
df = load_data()
df = preprocess(df)

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split into train/test using same random state for consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Run inference
def run_inference():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    print("Predictions on test set:")
    print(predictions)

    print("\nTrue values:")
    print(y_test.values)

if __name__ == "__main__":
    run_inference()
