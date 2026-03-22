import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "predict_flag_invoice.pkl"))
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"))

def predict_invoice_flag(input_data):
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    
    df = pd.DataFrame(input_data)
    
    if "total_item_quantity" in df.columns and "total_quantity" not in df.columns:
        df["total_quantity"] = df["total_item_quantity"]
        
    if "avg_receivingdelay" not in df.columns:
        df["avg_receivingdelay"] = 0.0
        
    features = [
        'invoice_quantity',
        'invoice_dollars',
        'Freight',
        'total_quantity',
        'total_item_dollars',
        'avg_receivingdelay'
    ]
    df_features = df[features]
    df_scaled = scaler.transform(df_features)
    
    predictions = model.predict(df_scaled)
    return {"Predicted_Flag": predictions}
