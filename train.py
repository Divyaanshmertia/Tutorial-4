import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess
from model import get_models
from evaluate import evaluate
import joblib
import os

df = load_data()
df = preprocess(df)

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = get_models()
results = []
os.makedirs("artifacts", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae, rmse, r2 = evaluate(y_test, preds)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    joblib.dump(model, f"artifacts/{name.replace(' ', '_')}.pkl")

results_df = pd.DataFrame(results)
results_df.to_csv("metrics.csv", index=False)

best_model = results_df.loc[results_df["RMSE"].idxmin()]
print(f"Best model: {best_model['model']} with RMSE: {best_model['RMSE']}")