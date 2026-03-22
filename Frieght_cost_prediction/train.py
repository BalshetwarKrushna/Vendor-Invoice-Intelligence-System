from pathlib import Path
import joblib

# Import modules
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
     # 1. Define paths clearly
    data_dir = Path("Data")
    db_path = data_dir / "inventory.db"
    model_dir = Path("models")

    # 2. ✅ CRITICAL: Create BOTH directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    print("Connected DB Path:", db_path)


    # Load data
    df = load_vendor_invoice_data()

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # ✅ Evaluate models (store results like in image)
    results = []

    results.append({
        "model_name": "Linear Regression",
        **evaluate_model(lr_model, X_test, y_test)
    })

    results.append({
        "model_name": "Decision Tree Regression",
        **evaluate_model(dt_model, X_test, y_test)
    })

    results.append({
        "model_name": "Random Forest Regression",
        **evaluate_model(rf_model, X_test, y_test)
    })

    print("\nModel Results:\n")
    for r in results:
        print(r)

    # ✅ Select best model (LOWEST MAE)
    best_model_info = min(results, key=lambda x: x["MAE"])
    best_model_name = best_model_info["model_name"]

    # Map name → model
    model_map = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = model_map[best_model_name]

    # ✅ Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()