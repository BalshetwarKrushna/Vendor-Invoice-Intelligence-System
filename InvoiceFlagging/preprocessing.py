import sqlite3
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Load Data
def load_data(db_path="Data/inventory.db"):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
    WITH purchase_agg AS (
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brand,
            SUM(p.Quantity) AS total_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receivingdelay
        FROM purchases p
        GROUP BY p.PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_pay_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brand,
        pa.total_quantity,
        pa.total_item_dollars,
        pa.avg_receivingdelay
    FROM vendor_invoice vi 
    LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
    """, conn)

    return df


# 2. Create Label Function
def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receivingdelay"] > 10:
        return 1
    return 0


# 3. Apply Labels
def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df



def split_data(df, features, target):

    X = df[features]
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )



def scale_features(X_train, X_test, scaler_path="models/scaler.pkl"):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    os.makedirs("models", exist_ok=True)

    
    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled