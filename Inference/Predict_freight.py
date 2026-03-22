import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "predict_freight_model.pkl"))

def predict_freight_cost(input_data):
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    
    if "Dollars" in input_data:
        df = pd.DataFrame({"Dollars": input_data["Dollars"]})
    else:
        df = pd.DataFrame(input_data)
        
    predictions = model.predict(df)
    return {"Predicted_Freight": predictions}
