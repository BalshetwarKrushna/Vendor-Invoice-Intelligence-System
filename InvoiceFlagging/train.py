from preprocessing import (
    load_data,
    apply_labels,
    split_data,
    scale_features
)

from model_evalution import (
    train_random_forest,
    evaluate_classifier
)

import joblib
import os


# Define features & target (GLOBAL like video)
FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_quantity',
    'total_item_dollars',
    'avg_receivingdelay'
]

TARGET = "flag_invoice"


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR,"..", "Data", "inventory.db")
    model_dir = os.path.join(BASE_DIR, "models")

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "predict_flag_invoice.pkl")

    # 🔹 Create models folder
    os.makedirs(model_dir, exist_ok=True)

    # 🔹 Load data
    df = load_data(data_path)
    df = apply_labels(df)

    # 🔹 Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "models/scaler.pkl"   # 👈 same as video
    )

    # 🔹 Train model
    grid_search = train_random_forest(X_train_scaled, y_train)

    # 🔹 Evaluate (IMPORTANT: best_estimator_)
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # 🔹 Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        grid_search.best_estimator_,
        "models/predict_flag_invoice.pkl"
    )


# Entry point
if __name__ == "__main__":
    main()