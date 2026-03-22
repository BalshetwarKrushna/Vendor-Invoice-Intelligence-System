import os
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

# 1. Load data
def load_vendor_invoice_data():
    base_dir = os.path.dirname(os.path.abspath("data/inventory.db"))  # current file location
    db_path = os.path.join(base_dir, "data", "inventory.db")

    print("DB Path:", "data/inventory.db")  # debug

    with sqlite3.connect("data/inventory.db") as conn:
        df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)

    return df


# 2. Prepare features
def prepare_features(df):
    # Feature engineering
    df['freight_per_unit'] = df['Freight'] / df['Quantity']

    # Input and target
    X = df[['Dollars']]   # you can later add more features
    y = df['Freight']

    return X, y


# 3. Split data
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
