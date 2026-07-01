# 🎯 E-Commerce Customer Churn Early Warning System

An end-to-end Machine Learning web application built to isolate high-risk retail consumer accounts before they churn. This project couples a **Random Forest Classifier** predictive engine with an interactive, production-ready corporate dashboard built in **Streamlit**.

---

## 💡 The Business Problem & Solution

### What We Are Solving
In e-commerce, acquiring a new customer is up to 5x more expensive than retaining an existing one. However, manually tracking which corporate accounts or shoppers are losing interest across millions of transaction lines is impossible. 

### How This App Solves It
This system acts as an automated monitoring pipeline. Business teams can export raw, unprocessed transaction data directly from their enterprise engine log and drop it straight into this dashboard. 

The application instantly:
1. Translates row-by-row shopping logs into aggregated customer profiles.
2. Extracts custom engineered metrics (Recency, Tenure, Average Order Value, etc.).
3. Feeds those metrics into a trained Machine Learning model to evaluate churn risk.
4. Outputs an executive risk score and triggers automated target action recommendations (`🚨 TARGET WITH PROMO` or `🟢 HEALTHY`).

---

## 🚀 App Architecture & Preprocessing Features

- **Dynamic Aggregation Pipeline:** Reads basic raw transaction data and automatically calculates critical features:
  - **Recency:** Number of days since the customer's last purchase.
  - **Tenure:** Number of days between the customer's very first and latest purchases.
  - **Frequency & Value:** Tracking total unique invoices, unique stock items bought, overall spend, and average order value.
- **Enterprise Asset Serialization:** Securely loads decoupled ML assets (`churn_random_forest_model.pkl` and `churn_data_scaler.pkl`) on runtime using memory-optimized data caching (`@st.cache_resource`).
- **Granular Data Search:** An interactive data table view allowing operational teams to search individual Customer IDs or filter entire risk cohorts at a glance.

---

## 📊 Expected CSV Input Data Schema

To test or run this dashboard, the uploaded CSV file **must match standard raw transaction logging structures** and include the following **8 columns** exactly as written below:

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| **Invoice** | Text / Int | Unique 6-digit transaction number | `580001` |
| **StockCode** | Text | Unique product/item identifier | `22423` |
| **Description** | Text | Human-readable item description | `REGENCY CAKESTAND 3 TIER` |
| **Quantity** | Integer | Total units purchased in the transaction line | `3` |
| **InvoiceDate** | Timestamp | Date/time of purchase (`YYYY-MM-DD HH:MM`) | `2026-06-29 10:30` |
| **Price** | Float | Individual single unit item cost | `12.75` |
| **Customer ID** | Text / Int | Unique customer account number | `99001` |
| **Country** | Text | Billing/shipping location country name | `United Kingdom` |

---

## 🛠️ Local Installation & Quick Start Guide

Want to run this full-stack predictive dashboard environment locally on your machine? Execute these commands sequentially in your terminal:

```bash
# 1. Clone the project repository from GitHub
git clone [https://github.com/mrinalepgp/ecommerce-customer-churn-analysis.git](https://github.com/mrinalepgp/ecommerce-customer-churn-analysis.git)

# 2. Navigate inside the root project directory
cd ecommerce-customer-churn-analysis

# 3. Install all required core pipeline application dependencies
pip install streamlit pandas numpy scikit-learn joblib

Once the launcher completes its execution trace, a local network web server will spin up and automatically open your dashboard application inside a fresh browser tab at http://localhost:8501. Enjoy exploring the analytical models!

# 4. Launch the local Streamlit web application environment
streamlit run app.py
