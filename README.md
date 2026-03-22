# 📦 Vendor Invoice Intelligence System

An **AI & Machine Learning-based system** to analyze vendor invoices, predict freight costs, and detect risky invoices — helping reduce **financial leakage, fraud risk, and manual workload**.

---

## 🚀 Project Overview

Organizations process thousands of invoices manually, leading to:

- ❌ Overpayments  
- ❌ Duplicate invoices  
- ❌ Incorrect freight charges  
- ❌ Fraud & audit risks  

This project solves these problems using **Machine Learning**:

- ✅ Predict expected freight cost  
- ✅ Detect abnormal or risky invoices  
- ✅ Enable data-driven financial decisions  
- ✅ Automate invoice validation  

---

## 🎯 Business Objectives

### 1️⃣ Freight Cost Prediction (Regression)

**Objective:**  
Predict expected freight cost using:
- Quantity  
- Invoice value  
- Historical patterns  

**Why it matters:**
- Freight is a major cost component  
- Poor estimation affects budgeting & margins  
- Improves vendor negotiation  

---

### 2️⃣ Invoice Risk Flagging (Classification)

**Objective:**  
Predict whether an invoice should be flagged for **manual approval**

**Why it matters:**
- Manual validation does not scale  
- High-value invoices carry higher risk  
- Early detection reduces fraud and audit issues  

---

## 🧠 End-to-End Pipeline
- Vendor Invoice Data
- ↓
- Data Cleaning & Preprocessing
- ↓
- Feature Engineering (SQL + Python)
- ↓
- Exploratory Data Analysis (EDA)
- ↓
- Machine Learning Models
- ↓
- Freight Cost Prediction + Risk Detection
  
---

## 📂 Data Sources

Data is stored in **SQLite database (`inventory.db`)**:

- `vendor_invoice` → Invoice-level data  
- `purchases` → Item-level details  
- `purchase_prices` → Reference pricing  
- `begin_inventory`, `end_inventory` → Inventory snapshots  

👉 SQL aggregation is used to generate invoice-level features.

---

## ⚙️ Key Features

- 📊 Data preprocessing & cleaning  
- 📈 Exploratory Data Analysis (EDA)  
- 🧮 Feature engineering using SQL + Python  
- 🤖 Regression model (Freight Prediction)  
- ⚠️ Classification model (Risk Detection)  
- 📉 Model evaluation using industry metrics  
- 🖥️ Streamlit dashboard for real-time prediction  

---

## 🛠️ Tech Stack

### 💻 Programming
- Python  

### 📊 Data Analysis
- Pandas  
- NumPy  

### 🗄️ Database
- SQLite  
- SQL  

### 🤖 Machine Learning
- Scikit-learn  
- Linear Regression  
- Decision Tree  
- Random Forest  

### 📈 Visualization
- Matplotlib  
- Seaborn  

### 📐 Statistics
- Hypothesis Testing  
- Cost Pattern Analysis  

### 🧰 Tools
- Jupyter Notebook  
- Git & GitHub  
- Streamlit  

---

## 🤖 Models Used

### 🔹 Regression (Freight Prediction)
- Linear Regression ✅ (Best Model)
- Decision Tree Regressor  
- Random Forest Regressor  

### 🔹 Classification (Invoice Flagging)
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier (with GridSearchCV)

---

## 📊 Model Evaluation

### 🔸 Regression Metrics
- RMSE  
- MAE  
- R² Score  

### 🔸 Classification Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Classification Report  

---

## 📈 Model Comparison & Results

### 🏆 Best Model: **Linear Regression**

### 📌 Key Findings:
- Strong **linear relationship** between:
  - Quantity ↔ Freight Cost  
  - Invoice Value ↔ Freight Cost  
- Linear Regression:
  - More stable  
  - Better generalization  
- Tree models showed higher variance  

---

## 📊 Data Insights & Visualization

### 🔹 Feature Relationship

![Freight Relationship](./images/download%201.png)

### 📌 Observations
- Strong positive correlation  
- Mostly linear trend  

---

### 🔹 Prediction vs Actual

![Prediction vs Actual](./images/download%202.png)

### 📌 Observations
- Predictions close to actual values  
- Few outliers  

---

## 🧠 Final Conclusion

- Freight cost depends strongly on quantity & invoice value  
- Linear Regression performs best for this dataset  
- System provides:
  - ✅ Accurate predictions  
  - ✅ Reliable anomaly detection  
  - ✅ Real-world financial impact  

---

## 🖥️ Streamlit Application

The project includes a **live dashboard**:

- Input invoice details  
- Predict freight cost instantly  
- Detect risky invoices  
- Display results in real-time  

---

## 🗂️ Project Structure
MachineLearningProj1/
│
├── data/
│ └── inventory.db
│
├── freight_cost_prediction/
├── invoice_flagging/
│
├── inference/
│ ├── predict_freight.py
│ └── predict_invoice_flag.py
│
├── models/
│ ├── predict_freight_model.pkl
│ ├── predict_flag_invoice.pkl
│ └── scaler.pkl
│
├── notebooks/
│
├── app.py
├── README.md

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
### 2️⃣ Install Dependencies
pip install -r requirements.txt
### 3️⃣ Train Models
python freight_cost_prediction/train.py
python invoice_flagging/train.py
4️⃣ Run Streamlit App
## 📊 Application UI Preview

### 🔹 Home Dashboard
![Dashboard](./images/Screenshot%908.png)

### 🔹 Freight Cost Prediction
![Freight Prediction](./images/Screenshot%910.png)

### 🔹 Invoice Risk Detection
![Invoice Risk](./images/Screenshot%909.png)
