import streamlit as st
import pickle
import joblib
import os
import requests
import numpy as np
import datetime as dt
import pandas as pd
import base64
from streamlit_lottie import st_lottie   # For animations

# =====================================================================
# MODEL LOADING
# =====================================================================
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=12Oe2aMIpZaAxir2uhGubtiU28Lh_T59_"
    response = requests.get(url)
    with open("model.joblib", "wb") as f:
        f.write(response.content)
    model = joblib.load("model.joblib")
    return model

model = load_model()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")

# =====================================================================
# BACKGROUND SETUP
# =====================================================================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: bottom center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

image_path = os.path.join(BASE_DIR, "Images", "logistics3.jpg")
set_bg(image_path)

st.title("How Punctual Is Your Delivery? Let's Check!")

# =====================================================================
# CLEAR BUTTON FUNCTIONALITY
# =====================================================================
def clear_inputs():
    for key in st.session_state.keys():
        st.session_state[key] = None

# =====================================================================
# LOTTIE LOADER
# =====================================================================
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Permanent logistics animations (verified)
EARLY_ANIM = "https://assets7.lottiefiles.com/packages/lf20_1pxqjqps.json"
ONTIME_ANIM = "https://assets4.lottiefiles.com/packages/lf20_j1adxtyb.json"
LATE_ANIM = "https://assets1.lottiefiles.com/packages/lf20_4DLPlK.json"


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    st.set_page_config(page_title="Order Delivery Prediction", layout="centered")
    st.markdown("Fill in the order details below to check delivery likelihood:")

    # Load data
    state_city_df = pd.read_csv(os.path.join(data_dir, "state_city_pairs.csv"))
    category_list = pd.read_csv(os.path.join(data_dir, "categories.csv"), header=None)[0].sort_values().unique().tolist()
    state_list = sorted(state_city_df['order_state'].dropna().unique())
    dept_list = pd.read_csv(os.path.join(data_dir, "departments.csv"), header=None)[0].sort_values().unique().tolist()
    cust_state_list = pd.read_csv(os.path.join(data_dir, "cust_states.csv"), header=None)[0].sort_values().unique().tolist()

    col1, col2 = st.columns(2)

    # ================================================================
    # INPUTS COLUMN 1
    # ================================================================
    with col1:
        payment_type = st.selectbox("Payment Type", ["-- Select --"] + ['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER'], index=0, key="payment_type")
        category_name = st.selectbox("Category Name", ["-- Select --"] + category_list, index=0, key="category_name")
        selected_state = st.selectbox("Select Order State", ["-- Select --"] + state_list, index=0, key="order_state")

        city_list = []
        if selected_state != "-- Select --":
            city_list = state_city_df[state_city_df['order_state'] == selected_state]['order_city']
            city_list = city_list.dropna().sort_values().unique().tolist()

        selected_city = st.selectbox("Select Order City", ["-- Select --"] + city_list, index=0, key="order_city")
        customer_state = st.selectbox("Product Location", ["-- Select --"] + cust_state_list, index=0, key="customer_state")
        department_name = st.selectbox("Department Name", ["-- Select --"] + dept_list, index=0, key="department_name")
        market = st.selectbox("Market", ["-- Select --"] + ['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'], index=0, key="market")

    # ================================================================
    # INPUTS COLUMN 2
    # ================================================================
    with col2:
        order_item_quantity = st.number_input("Order Item Quantity", min_value=0.0, format="%.1f", key="order_item_quantity")
        order_item_discount = st.number_input("Order Item Discount ($)", min_value=0.0, format="%.2f", key="order_item_discount")
        order_item_total_amount = st.number_input("Order Item Total Amount ($)", min_value=0.0, format="%.2f", key="order_item_total_amount")

        order_status = st.selectbox("Shipping Status",
                                    ["-- Select --"] + ['CLOSED', 'COMPLETE', 'ON_HOLD', 'PAYMENT_REVIEW',
                                                        'PENDING', 'PENDING_PAYMENT', 'PROCESSING'],
                                    index=0, key="order_status")

        shipping_mode = st.selectbox("Shipping Mode",
                                     ["-- Select --"] + ['First Class', 'Second Class', 'Same Day', 'Standard Class'],
                                     index=0, key="shipping_mode")

        order_date = st.date_input("Order Date", value=None, key="order_date")
        ship_date = st.date_input("Ship Date", value=None, key="ship_date")

    # CLEAR BUTTON
    st.button("Clear", on_click=clear_inputs)

    # VALIDATION
    if None in [
        payment_type, category_name, selected_state, selected_city,
        customer_state, department_name, market, order_item_quantity,
        order_item_discount, order_item_total_amount, order_status,
        shipping_mode, order_date, ship_date
    ] or "-- Select --" in [
        payment_type, category_name, selected_state, selected_city,
        customer_state, department_name, market, order_status, shipping_mode
    ]:
        st.warning("Please fill in all fields before predicting.")
        return

    # DATE CHECK
    date_diff = (ship_date - order_date).days
    if date_diff < 0:
        st.error("⚠️ Ship Date cannot be before Order Date!")
        return

    # LABEL ENCODERS
    def label_map(values_list):
        return {v: i for i, v in enumerate(sorted(values_list))}

    mappings = {
        'payment_type': label_map(['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER']),
        'order_status': label_map(['CLOSED', 'COMPLETE', 'ON_HOLD', 'PAYMENT_REVIEW', 'PENDING', 'PENDING_PAYMENT', 'PROCESSING']),
        'shipping_mode': label_map(['First Class', 'Same Day', 'Second Class', 'Standard Class']),
        'market': label_map(['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA']),
        'category_name': label_map(category_list),
        'order_city': label_map(state_city_df['order_city'].dropna().unique().tolist()),
        'order_state': label_map(state_city_df['order_state'].dropna().unique().tolist()),
        'department_name': label_map(dept_list),
        'customer_state': label_map(cust_state_list)
    }

    # FEATURE VECTOR
    features = np.array([[
        mappings['payment_type'][payment_type],
        mappings['category_name'][category_name],
        mappings['order_city'][selected_city],
        order_item_quantity,
        order_item_total_amount,
        mappings['order_state'][selected_state],
        mappings['order_status'][order_status],
        mappings['shipping_mode'][shipping_mode],
        date_diff,
        mappings['market'][market],
        mappings['customer_state'][customer_state],
        order_item_discount,
        mappings['department_name'][department_name]
    ]])

    # ================================================================
    # PREDICTION + ANIMATION
    # ================================================================
    if st.button("PREDICT"):
        result = model.predict(features)
        pred_class = result[0]

        # Alerts
        if pred_class == 0:
            st.success("The order is likely to be delivered **early**")
        elif pred_class == 1:
            st.info("The order is likely to be delivered **on time**")
        else:
            st.error("The order is likely to be **late**")

        # Status text
        labels = ["Early", "On Time", "Late"]
        st.subheader(f"Delivery Status: {labels[pred_class]}")

        # Pick animation
        if pred_class == 0:
            anim_url = EARLY_ANIM
        elif pred_class == 1:
            anim_url = ONTIME_ANIM
        else:
            anim_url = LATE_ANIM

        animation = load_lottie(anim_url)

        if animation:
            st_lottie(animation, height=250)
        else:
            st.info("🚚 Animation could not load.")


# RUN APP
if __name__ == "__main__":
    main()

