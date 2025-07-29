# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load trained model and encoders ===
model = joblib.load("best_model.pkl")
le_store = joblib.load("store_encoder.pkl")
le_product = joblib.load("product_encoder.pkl")
final_features = joblib.load("final_features.pkl")

# === Load the dataset ===
df = pd.read_csv("retail_store_inventory.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# === UI ===
st.title("🛍️ Retail Sales Predictor")
st.write("Predict units sold for a product in a given store and month.")

# === User Inputs ===
year = st.selectbox("Select Year", list(range(2023, 2026)))  # Allows 2022 to 2026
month = st.selectbox("Select Month", sorted(df['Month'].unique()))
store_id = st.selectbox("Select Store ID", sorted(df['Store ID'].unique()))
product_id = st.selectbox("Select Product ID", sorted(df['Product ID'].unique()))

# === Prediction Logic ===
if st.button("Predict Sales"):
    try:
        store_enc = le_store.transform([store_id])[0]
        product_enc = le_product.transform([product_id])[0]
    except ValueError:
        st.error("❌ Store or Product not found in training data")
        st.stop()

    subset = df[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
    if subset.empty:
        st.error("❌ No past data available for this Store/Product combination")
        st.stop()

    input_data = {
        'Year': year,
        'Month': month,
        'Store_enc': store_enc,
        'Product_enc': product_enc,
        'Inventory Level': subset['Inventory Level'].mean(),
        'Price': subset['Price'].mean(),
        'Discount': subset['Discount'].mean()
    }

    # Add additional features if required by model
    if 'Holiday_Season' in final_features:
        input_data['Holiday_Season'] = 1 if month in [11, 12] else 0
    if 'Prev_Units_Sold' in final_features:
        input_data['Prev_Units_Sold'] = subset['Units Sold'].iloc[-1] if not subset.empty else df['Units Sold'].mean()

    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df[final_features])[0]

    st.success(f"📦 Predicted Units Sold for {product_id} at {store_id} in {month}-{year}: {prediction:.2f}")

    # Inventory Suggestion
    avg_units = df['Units Sold'].mean()
    if prediction > avg_units:
        st.info("📈 Suggestion: Stock More")
    elif prediction < 0.9 * avg_units:
        st.warning("📉 Suggestion: Stock Less")
    else:
        st.success("✅ Suggestion: Maintain Stock Level")
