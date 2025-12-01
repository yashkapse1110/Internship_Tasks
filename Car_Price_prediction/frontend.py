import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

y_test = np.load("y_test.npy")
y_pred = np.load("y_pred.npy")

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction",
                   page_icon="🚗",
                   layout="wide")

# Title Section
st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>🚗 Car Price Prediction System</h1>
    <h4 style='text-align:center; color:gray;'>Powered by Machine Learning (Random Forest)</h4>
""", unsafe_allow_html=True)
st.write("---")

# Sidebar - User Input
st.sidebar.header("Enter Car Details")
Present_Price = st.sidebar.number_input("Present Price (in Lakhs)", value=5.0)
Driven_kms = st.sidebar.number_input("Driven Kilometers", value=20000)

Fuel_Type = st.sidebar.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
Selling_type = st.sidebar.selectbox("Selling Type", ("Dealer", "Individual"))
Transmission = st.sidebar.selectbox("Transmission", ("Manual", "Automatic"))
Owner = st.sidebar.number_input("Number of Previous Owners", min_value=0, max_value=3, value=0)
Year = st.sidebar.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)

# Convert categorical values
fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
selling_map = {"Individual":0, "Dealer":1}
trans_map = {"Manual":0, "Automatic":1}

Car_Age = 2025 - Year

# ---------------------------
# Prediction Button
# ---------------------------
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame([[
        Present_Price,
        Driven_kms,
        fuel_map[Fuel_Type],
        selling_map[Selling_type],
        trans_map[Transmission],
        Owner,
        Car_Age
    ]], columns=[
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age"
    ])

    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 **Estimated Selling Price: ₹ {round(prediction, 2)} Lakhs**")
    st.write("---")

# ---------------------------
# Graphs Section
# ---------------------------
st.subheader("📊 Model Insights & Visualization")

try:
    pass
except:
    st.info("Graphs will appear after integrating test prediction arrays.")
    st.stop()


# ---------------------------
# 1) Actual vs Predicted Graph
# ---------------------------
st.write("### 📈 Actual vs Predicted Price")

fig1 = plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Price")
plt.grid(True)

st.pyplot(fig1)

# ---------------------------
# 2) Error Distribution
# ---------------------------
st.write("### 📉 Error Distribution")

errors = y_test - y_pred

fig2 = plt.figure(figsize=(6,4))
plt.hist(errors, bins=15)
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Prediction Error Distribution")
plt.grid(True)

st.pyplot(fig2)

# ---------------------------
# 3) Feature Importance
# ---------------------------
st.write("### ⭐ Feature Importance")

importances = model.feature_importances_
feature_names = ["Present_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner","Car_Age"]

fig3 = plt.figure(figsize=(6,4))
plt.barh(feature_names, importances)
plt.xlabel("Importance Score")
plt.title("Which Features Matter Most?")
plt.grid(True)

st.pyplot(fig3)
