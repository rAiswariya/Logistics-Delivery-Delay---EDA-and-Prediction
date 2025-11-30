import streamlit as st
import pickle
import joblib
import os
import requests
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
import base64

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
# Background setup
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
# RESET BUTTON FUNCTIONALITY
# =====================================================================
def clear_inputs():
    for key in st.session_state.keys():
        st.session_state[key] = None


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    st.set_page_config(page_title="Order Delivery Prediction", layout="centered")
    st.markdown("Fill in the order details below to check delivery likelihood:")

    state_city_df = pd.read_csv(os.path.join(data_dir, "state_city_pairs.csv"))
    category_list = pd.read_csv(os.path.join(data_dir, "categories.csv"), header=None)[0].sort_values().unique().tolist()
    state_list = sorted(state_city_df['order_state'].dropna().unique())
    dept_list = pd.read_csv(os.path.join(data_dir, "departments.csv"), header=None)[0].sort_values().unique().tolist()
    cust_state_list = pd.read_csv(os.path.join(data_dir, "cust_states.csv"), header=None)[0].sort_values().unique().tolist()

    col1, col2 = st.columns(2)

    with col1:
        payment_type = st.selectbox(
            "Payment Type",
            ["-- Select --"] + ['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER'],
            index=0,
            key="payment_type"
        )

        category_name = st.selectbox(
            "Category Name",
            ["-- Select --"] + category_list,
            index=0,
            key="category_name"
        )

        selected_state = st.selectbox(
            "Select Order State",
            ["-- Select --"] + state_list,
            index=0,
            key="order_state"
        )

        city_list = []
        if selected_state != "-- Select --":
            city_list = state_city_df[state_city_df['order_state'] == selected_state]['order_city']
            city_list = city_list.dropna().sort_values().unique().tolist()

        selected_city = st.selectbox(
            "Select Order City",
            ["-- Select --"] + city_list,
            index=0,
            key="order_city"
        )

        customer_state = st.selectbox(
            "Product Location",
            ["-- Select --"] + cust_state_list,
            index=0,
            key="customer_state"
        )

        department_name = st.selectbox(
            "Department Name",
            ["-- Select --"] + dept_list,
            index=0,
            key="department_name"
        )

        market = st.selectbox(
            "Market",
            ["-- Select --"] + ['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'],
            index=0,
            key="market"
        )

    with col2:
        order_item_quantity = st.number_input(
            "Order Item Quantity",
            min_value=0.0,
            format="%.1f",
            key="order_item_quantity"
        )

        order_item_discount = st.number_input(
            "Order Item Discount ($)",
            min_value=0.0,
            format="%.2f",
            key="order_item_discount"
        )

        order_item_total_amount = st.number_input(
            "Order Item Total Amount ($)",
            min_value=0.0,
            format="%.2f",
            key="order_item_total_amount"
        )

        order_status = st.selectbox(
            "Shipping Status",
            ["-- Select --"] + [
                'CLOSED', 'COMPLETE', 'ON_HOLD', 'PAYMENT_REVIEW',
                'PENDING', 'PENDING_PAYMENT', 'PROCESSING'
            ],
            index=0,
            key="order_status"
        )

        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["-- Select --"] + [
                'First Class', 'Second Class', 'Same Day', 'Standard Class'
            ],
            index=0,
            key="shipping_mode"
        )

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
        'order_status': label_map([
            'CLOSED', 'COMPLETE', 'ON_HOLD', 'PAYMENT_REVIEW',
            'PENDING', 'PENDING_PAYMENT', 'PROCESSING'
        ]),
        'shipping_mode': label_map(['First Class', 'Same Day', 'Second Class', 'Standard Class']),
        'market': label_map(['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA']),
        'category_name': label_map(category_list),
        'order_city': label_map(state_city_df['order_city'].dropna().unique().tolist()),
        'order_state': label_map(state_city_df['order_state'].dropna().unique().tolist()),
        'department_name': label_map(dept_list),
        'customer_state': label_map(cust_state_list)
    }

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

    if st.button("PREDICT"):
        result = model.predict(features)

        if result[0] == 0:
            st.success("The order is likely to be delivered **early**")
        elif result[0] == 1:
            st.info("The order is likely to be delivered **on time**")
        else:
            st.error("The order is likely to be **late**")

        probs = model.predict_proba(features)[0]
        labels = ['Early', 'On Time', 'Late']
        pred_class = result[0]
        pred_prob = probs[pred_class] * 100

        colors = {
            0: ("green", [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "green"},
                {'range': [80, 100], 'color': "darkgreen"}
            ]),
            1: ("blue", [
                {'range': [0, 50], 'color': "lightblue"},
                {'range': [50, 80], 'color': "deepskyblue"},
                {'range': [80, 100], 'color': "navy"}
            ]),
            2: ("red", [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 80], 'color': "tomato"},
                {'range': [80, 100], 'color': "darkred"}
            ])
        }

        bar_color, steps = colors[pred_class]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob,
            title={'text': f"Prediction Confidence: {labels[pred_class]}"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': bar_color}, 'steps': steps}
        ))

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
