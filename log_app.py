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

def set_bg(image_file):

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Background image setup */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;            /* Scale to fill */
            background-position: bottom center; /* Keep bottom visible */
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Semi-transparent overlay for readability */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
        }}

        /* Make all text white for contrast */
        .block-container, .block-container * {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

image_path = os.path.join(BASE_DIR, "Images", "logistics3.jpg")
set_bg(image_path)

st.title("How Punctual Is Your Delivery? Let's Check!")

def main():
    st.set_page_config(page_title="Order Delivery Prediction", layout="centered")
    st.markdown("Fill in the order details below to check delivery likelihood:")

    state_city_df = pd.read_csv(os.path.join(data_dir, "state_city_pairs.csv"))
    category_list = pd.read_csv(os.path.join(data_dir, "categories.csv"), header=None)[
        0].sort_values().unique().tolist()
    state_list = sorted(state_city_df['order_state'].dropna().unique())
    dept_list = pd.read_csv(os.path.join(data_dir, "departments.csv"), header=None)[0].sort_values().unique().tolist()
    cust_state_list = pd.read_csv(os.path.join(data_dir, "cust_states.csv"), header=None)[
        0].sort_values().unique().tolist()
    # ==========================
    # User Inputs
    # ==========================
    col1, col2 = st.columns(2)
    with col1:
        payment_type = st.selectbox("Payment Type", ['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        category_name = st.selectbox("Category Name", category_list)
        selected_state = st.selectbox("Select Order State", state_list)
        city_list = (state_city_df[state_city_df['order_state'] == selected_state][
                         'order_city'].dropna().sort_values().unique().tolist())
        selected_city = st.selectbox("Select Order City", city_list)
        customer_state = st.selectbox("Product Location", cust_state_list)
        department_name = st.selectbox("Department Name", dept_list)
        market = st.selectbox("Market", ['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA'])

    with col2:
        order_item_quantity = st.number_input("Order Item Quantity", min_value=1.0)
        order_item_discount = st.number_input("Order Item Discount ($)", min_value=0.0)
        order_item_total_amount = st.number_input("Order Item Total Amount ($)", min_value=0.0)
        order_status = st.selectbox("Shipping Status", [
            'CLOSED', 'COMPLETE', 'ON_HOLD', 'PAYMENT_REVIEW',
            'PENDING', 'PENDING_PAYMENT', 'PROCESSING'
        ])
        shipping_mode = st.selectbox("Shipping Mode", [
            'First Class', 'Second Class', 'Same Day', 'Standard Class'
        ])
        order_date = st.date_input("Order Date", dt.date(2024, 1, 1))
        ship_date = st.date_input("Ship Date", dt.date(2024, 1, 2))

    # ==========================
    # Compute date difference
    # ==========================
    date_diff = (ship_date - order_date).days
    if date_diff < 0:
        st.error("⚠️ Ship Date cannot be before Order Date!")

    # ==========================
    # Label Encoding (alphabetical)
    # ==========================
    # ==========================
    # Label Encoding (alphabetical)
    # ==========================
    def label_map(values_list):
        return {v: i for i, v in enumerate(sorted(values_list))}

    # Create label encodings using the available lists
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

    # Apply mappings
    payment_value = mappings['payment_type'][payment_type]
    order_status_value = mappings['order_status'][order_status]
    shipping_mode_value = mappings['shipping_mode'][shipping_mode]
    market_value = mappings['market'][market]
    category_value = mappings['category_name'][category_name]
    city_value = mappings['order_city'][selected_city]
    state_value = mappings['order_state'][selected_state]
    dept_value = mappings['department_name'][department_name]
    cust_state_value = mappings['customer_state'][customer_state]

    # ==========================
    # Create final feature vector
    # ==========================
    features = np.array([[
        payment_value, category_value, city_value, order_item_quantity,
        order_item_total_amount, state_value, order_status_value,
        shipping_mode_value, date_diff, market_value, cust_state_value,
        order_item_discount, dept_value
    ]])

    if st.button("PREDICT"):
        result = model.predict(features)

        if result[0] == 0:
            st.success(f"The order is likely to be delivered **early**")
        elif result[0] == 1:
            st.info(f"The order is likely to be delivered **on time**")
        else:
            st.error(f"The order is likely to be **late**")


        probs = model.predict_proba(features)[0]
        pred_class = result[0]
        labels = ['Early', 'On Time', 'Late']
        pred_label = labels[pred_class]

        # Get the probability for the predicted label
        pred_prob = probs[pred_class] * 100  # convert to percentage

        # Define gauge colors and range logic
        if pred_class == 0:  # Early
            bar_color = "green"
            steps = [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "green"},
                {'range': [80, 100], 'color': "darkgreen"}
            ]
        elif pred_class == 1:  # On Time
            bar_color = "blue"
            steps = [
                {'range': [0, 50], 'color': "lightblue"},
                {'range': [50, 80], 'color': "deepskyblue"},
                {'range': [80, 100], 'color': "navy"}
            ]
        else:  # Late
            bar_color = "red"
            steps = [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 80], 'color': "tomato"},
                {'range': [80, 100], 'color': "darkred"}
            ]

        # Create the gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob,
            title={'text': f"Prediction Confidence: {pred_label}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},
                'steps': steps
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()