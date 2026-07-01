# Customer Churn Early Warning System

An end-to-end Machine Learning web application that predicts e-commerce customer churn using a **Random Forest Classifier** wrapped in a **Streamlit** dashboard.

## 🚀 Features
- **Automated Pipeline:** Automatically calculates user aggregation data (Recency, Tenure, Order Count, Spend Metrics) dynamically from a raw 8-column store transaction log.
- **Enterprise Ready:** Custom data preprocessing pipeline featuring standalone model serialization (`joblib`) and automated `StandardScaler` synchronization.
- **Interactive Filtering:** Live business intelligence dashboard allowing operators to filter high-risk cohorts instantly.
## 📊 Expected CSV Data Schema

When running the Streamlit app, the uploaded CSV must contain the following **8 raw columns** (exactly matching standard transaction engine logs):

| Column Name | Description | Example Value |
| :--- | :--- | :--- |
| **Invoice** | Unique 6-digit transaction number | `580001` |
| **StockCode** | Unique product item identifier | `22423` |
| **Description** | Product name text | `REGENCY CAKESTAND 3 TIER` |
| **Quantity** | Total unit count purchased | `3` |
| **InvoiceDate** | Timestamp of purchase (`YYYY-MM-DD HH:MM`) | `2026-06-29 10:30` |
| **Price** | Product unit cost | `12.75` |
| **Customer ID** | Unique customer account number | `99001` |
| **Country** | Name of shopper's country | `United Kingdom` |
