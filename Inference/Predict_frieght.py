import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(
    BASE_DIR,
    "..",
    "InvoiceFlagging",
    "models",
    "predict_freight_model.pkl"
))


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice(input_data):

    model = load_model()

    # Input dataframe (ONLY Dollars)
    df = pd.DataFrame(input_data)

    n = len(df)

    # Create model input (1 feature)
    model_input = pd.DataFrame({
        "Dollars": df["Dollars"]
    })

    # Predict
    predictions = model.predict(model_input)

    # 🔥 FINAL OUTPUT (LIKE VIDEO)
    output_df = pd.DataFrame({
        "Dollars": df["Dollars"],
        "Prediction": predictions
    })

    return output_df


if __name__ == "__main__":

    sample_input = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    result = predict_invoice(sample_input)
    print(result)