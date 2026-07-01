import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up clean page configuration
st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")

st.title("🎯 Customer Churn Early Warning System")
st.markdown("Upload raw transaction history files to instantly isolate high-risk customer profiles.")

# 1. Safely load our frozen AI components (Cleaned for Production)
@st.cache_resource
def load_assets():
    import os
    all_files = os.listdir('.')
    
    # Automatically hunt for the model and scaler file names in the folder
    model_file = [f for f in all_files if 'churn_random_forest_model' in f][0]
    scaler_file = [f for f in all_files if 'churn_data_scaler' in f][0]
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

# 2. Try loading the assets and handle the sidebar display cleanly
try:
    model, scaler = load_assets()
    st.sidebar.success("🧠 AI Engine Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading files: {str(e)}")
    
# 3. Build the File Upload interface
uploaded_file = st.file_uploader("Drop your raw transaction CSV file here:", type=["csv"])

if uploaded_file is not None:
    # Read the raw data uploaded by the user (accepts the original 8 columns)
    df_raw = pd.read_csv(uploaded_file)
    st.success("📄 Raw transaction data loaded successfully!")
    
    with st.spinner("Processing transaction profiles..."):
        # Format dates directly from the raw data
        df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
        snapshot_date = df_raw['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Calculate line item totals dynamically from Quantity and Price columns
        df_raw['Total_Cost'] = df_raw['Quantity'] * df_raw['Price']
        
        # Build the 6 feature profile matrix seamlessly
        customer_df = df_raw.groupby('Customer ID').agg({
            'InvoiceDate': [lambda x: (snapshot_date - x.max()).days, lambda x: (x.max() - x.min()).days],
            'Invoice': 'nunique',
            'StockCode': 'nunique',
            'Total_Cost': 'sum',
            'Country': lambda x: 1 if x.iloc[0] != 'United Kingdom' else 0
        })
        
        customer_df.columns = ['Recency_Days', 'Customer_Tenure_Days', 'Total_Orders', 'Unique_Items_Bought', 'Total_Spent', 'Is_International']
        customer_df['Avg_Order_Value'] = customer_df['Total_Spent'] / customer_df['Total_Orders']
        
        # Isolate features for the model
        feature_cols = ['Customer_Tenure_Days', 'Total_Orders', 'Unique_Items_Bought', 'Total_Spent', 'Is_International', 'Avg_Order_Value']
        X_new = customer_df[feature_cols]
        
        # Scale the new data using our production scaler asset
        X_new_scaled = scaler.transform(X_new)
        
        # 4. Generate Blind Predictions and Probability Metrics
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)[:, 1] # Risk percentage
        
        # Append predictions back to a clean user-facing table
        output_df = customer_df.copy().reset_index()
        output_df['Churn_Risk_Score'] = (probabilities * 100).round(1)
        output_df['System_Action_Flag'] = np.where(predictions == 1, "🚨 TARGET WITH PROMO", "🟢 HEALTHY")
        
    # 5. Display Corporate Metrics
    st.header("📋 Executive Summary Risk Analysis")
    col1, col2, col3 = st.columns(3)
    
    total_customers = output_df.shape[0]
    flagged_churners = int((predictions == 1).sum())
    projected_churn_rate = (flagged_churners / total_customers) * 100
    
    col1.metric("Total Profiles Analyzed", f"{total_customers:,}")
    col2.metric("At-Risk Accounts Flagged", f"{flagged_churners:,}")
    col3.metric("Predicted Next-Month Churn Rate", f"{projected_churn_rate:.1f}%")
    
    # 6. Interactive Risk Filter Data Table
    st.subheader("🔍 Deep Dive: Search Specific Customer IDs")
    filter_status = st.selectbox("Filter Profiles By Status Flag:", ["Show All", "🚨 TARGET WITH PROMO", "🟢 HEALTHY"])
    
    if filter_status != "Show All":
        display_df = output_df[output_df['System_Action_Flag'] == filter_status]
    else:
        display_df = output_df
        
    st.dataframe(display_df[['Customer ID', 'System_Action_Flag', 'Churn_Risk_Score', 'Total_Orders', 'Total_Spent', 'Avg_Order_Value']].sort_values(by='Churn_Risk_Score', ascending=False))