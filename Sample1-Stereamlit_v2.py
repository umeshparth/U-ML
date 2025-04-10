import streamlit as st
import pandas as pd
import os
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration with fixed height to avoid scrolling
st.set_page_config(page_title="Bank Loan Approval Prediction", layout="wide", initial_sidebar_state="expanded")

# Load model and scaler
try:
    if not os.path.exists('models/best_model.pkl') or not os.path.exists('models/scaler.pkl'):
        st.error("Model or scaler files not found! Please run 'train_model.py' first.")
        st.stop()
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Custom CSS with larger fonts and no header
st.markdown("""
    <style>
    .stApp {
        background-color: #e8ecef;
        font-family: 'Arial', sans-serif;
        max-height: 100vh;
        overflow: hidden;
    }
    .simple-text {
        background-color: #362479;
        color: #ffffff;
        padding: 15px 20px;
        text-align: center;
        font-family: 'Georgia', serif;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
        margin-bottom: 10px;
        min-height: 70px;
        width: 100%;
        z-index: 1;
    }
    .simple-text h1 {
        font-size: 36px; /* Very large font */
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .simple-text p {
        font-size: 24px; /* Very large font */
        margin: 2px 0 0 0;
        font-style: italic;
        opacity: 0.9;
    }
    .stButton>button {
        background-color: #362479;
        color: white;
        border-radius: 6px;
        font-size: 22px; /* Very large font */
        padding: 10px 20px;
        width: 100%;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: ##241852;
    }
        .title {
        font-size: 32px;
        color: #ffffff; /* Bank-like navy blue */
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }    
    .title {
        font-size: 32px;
        color: #362479; /* Bank-like navy blue */
        text-align: left;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        text-align: left;
        margin-bottom: 20px;
    }        
    .stTextInput, .stNumberInput, .stSelectbox {
        border: 1px solid #b0c4de;
        border-radius: 4px;
        padding: 8px; /* Larger padding */
        margin: 2px 0;
        background-color: #ffffff;
        font-size: 18px; /* Very large font */
    }
    .input-group {
        background-color: #f0f4f8;
        padding: 12px; /* Larger padding */
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .input-group > div {
        font-size: 22px; /* Very large font */
        color: #2e7d7d;
    }
   /* Animated Outcome with status-based colors */
    .animated-outcome {
        border-radius: 6px;
        padding: 15px;
        font-size: 24px;
        animation: highlight 2s ease-in-out infinite;
        text-align: center;
        margin-top: 10px;
    }
    .animated-outcome.approved {
        background-color: #e0f0e0;
        border: 2px solid #4CAF50;
        color: #4CAF50;
    }
    .animated-outcome.approved:hover {
        background-color: #c0e0c0;
        box-shadow: 0 0 10px #4CAF50;
    }
    .animated-outcome.not-approved {
        background-color: #ffe0e0;
        border: 2px solid #F44336;
        color: #F44336;
    }
    .animated-outcome.not-approved:hover {
        background-color: #ffcccc;
        box-shadow: 0 0 10px #F44336;
    }        
    @keyframes highlight {
        0% { background-color: #e0f0f0; box-shadow: 0 0 0 #2e7d7d; }
        50% { background-color: #c0e0e0; box-shadow: 0 0 10px #2e7d7d; }
        100% { background-color: #e0f0f0; box-shadow: 0 0 0 #2e7d7d; }
    }
    /* Custom Layout */
   .ui-column {
        width: 70% !important;
        padding: 0 10px;
        float: left;
    }
    .graph-column {
        width: 30% !important;
        max-width: 30% !important; /* Enforce maximum width */
        padding: 0 10px;
        float: right;
        overflow-y: auto;
        max-height: 80vh;
        box-sizing: border-box; /* Ensure padding doesn’t affect width */
    }
    .plotly-graph-div {
        height: 180px !important;
        width: 100% !important; /* Ensure full width within column */
        max-width: 100% !important; /* Prevent expansion beyond column */
        margin: 5px 0;
        box-sizing: border-box; /* Include padding/border in width */
    }
    /* Multi-column input layout */
    .input-row {
        display: flex;
        # Gauge Chart: Confidence Level with fixed width
         fig_gauge = go.Figure(go.Indicator(
             mode="gauge+number",
-            value=latest_prediction["confidence"] * 100,
+            value=latest_prediction["confidence"] * 100,
+            domain={'x': [0, 1], 'y': [0, 1]},
+            title={'text': "Confidence"},
+            gauge={'axis': {'range': [0, 100]},
+                   'bar': {'color': "#4a9a9a"},
+                   'steps': [
+                       {'range': [0, 50], 'color': "#d1e0e0"},
             domain={'x': [0, 1], 'y': [0, 1]},
             title={'text': "Confidence"},
             gauge={'axis': {'range': [0, 100]},    
            {'range': [75, 100], 'color': "#b0d0d0"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 60}},
             number={'font': {'size': 20}}
    }
    .input-row .stTextInput, .input-row .stNumberInput, .input-row .stSelectbox {
        flex: 1;
        margin-right: 5px;
        font-size: 18px; /* Very large font */
    }
    .input-row .stTextInput:last-child, .input-row .stNumberInput:last-child, .input-row .stSelectbox:last-child {
        margin-right: 0;
    }
    .loan-status {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="title">Bank Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Assess loan applications with precision and efficiency</div>', unsafe_allow_html=True)

# Sidebar for instructions or branding
with st.sidebar:
    st.image("UVRBank.png")  # Replace with actual bank logo URL
    st.write("### Instructions")
    st.write("Enter the applicant’s details below to predict loan approval status. Ensure all fields are filled accurately.")
    st.write("---")
    st.write("**Contact Support**: support@bankname.com")


# Main content with custom layout
ui_col, graph_col = st.columns([0.7, 0.2], gap="small")

with ui_col:
    # Compact UI with multi-column rows and new labels
    with st.container():
        with st.container():
            st.write("### Personal Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                full_name = st.text_input("Full Name", value="John Doe", help="Applicant’s full name")
            with col2:
                phone = st.text_input("Phone", value="9876543210", help="10-digit phone number")
            with col3:
                edu_level = st.selectbox("Education", ['Graduate', 'Non-Graduate'], help="Educational background")
            with col4:
                emp_status = st.selectbox("Employment", ['Yes', 'No'], help="Self-employed status")
        
        with st.container():
            st.write("### Loan Details")
            col1, col2 = st.columns(2)
            with col1:
                loan_value = st.number_input("Loan Value (INR)", min_value=0.0, value=200000.0, step=10000.0, format="%.2f", help="Loan amount")
            with col2:
                duration = st.number_input("Duration (Years)", min_value=1, max_value=30, value=10, help="Loan term")
            
        with st.container():
            st.write("### Financial Profile")
            col1, col2 = st.columns(2)
            with col1:
                yearly_income = st.number_input("Income (INR)", min_value=0.0, value=500000.0, step=10000.0, format="%.2f", help="Annual income")
            with col2:
                credit_rate = st.number_input("Credit Score", min_value=300, max_value=900, value=750, help="Credit rating")
            
        with st.container():
            st.write("### Asset Holdings")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                home_value = st.number_input("Home Value (INR)", min_value=0.0, value=300000.0, step=10000.0, format="%.2f", help="Residential value")
            with col2:
                bus_value = st.number_input("Business Value (INR)", min_value=0.0, value=150000.0, step=10000.0, format="%.2f", help="Commercial value")
            with col3:
                lux_value = st.number_input("Luxury Value (INR)", min_value=0.0, value=200000.0, step=10000.0, format="%.2f", help="Luxury assets")
            with col4:
                bank_funds = st.number_input("Bank Funds (INR)", min_value=0.0, value=100000.0, step=10000.0, format="%.2f", help="Bank assets")
            
        # Predict button
        if st.button("Predict Loan Status"):
            input_data = pd.DataFrame({
                'education': [1 if edu_level == 'Graduate' else 0],
                'self_employed': [1 if emp_status == 'Yes' else 0],
                'income_annum': [yearly_income],
                'loan_amount': [loan_value],
                'loan_term': [duration],
                'cibil_score': [credit_rate],
                'residential_assets_value': [home_value],
                'commercial_assets_value': [bus_value],
                'luxury_assets_value': [lux_value],
                'bank_asset_value': [bank_funds]
            })
            
            numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                             'residential_assets_value', 'commercial_assets_value',
                             'luxury_assets_value', 'bank_asset_value']
            
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of approval
            result = "Approved" if prediction == 1 else "Not Approved"
            
            # Store prediction history
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:M:S"),
                'name': full_name,
                'status': result,
                'confidence': prediction_proba
            })

with graph_col:
    # Loan Status at the top
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        latest_prediction = st.session_state.prediction_history[-1]
        status_class = "animated-outcome approved" if latest_prediction["status"] == "Approved" else "animated-outcome not-approved"
        st.markdown(f'<div class="{status_class} loan-status">Loan Status: <b>{latest_prediction["status"]}</b><br>Confidence: <b>{latest_prediction["confidence"]:.2%}</b></div>', unsafe_allow_html=True)
        if latest_prediction["status"] == "Approved":
            st.write("**Note**: Subject to final verification.")
        else:
            st.write("**Note**: Review details for improvement.")

        # Gauge Chart: Confidence Level
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_prediction["confidence"] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#4a9a9a"},
                   'steps': [
                       {'range': [0, 50], 'color': "#d1e0e0"},
                       {'range': [50, 75], 'color': "#c0d8d8"},
                       {'range': [75, 100], 'color': "#b0d0d0"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'value': 60}},
            number={'font': {'size': 20}}
        ))
        fig_gauge.update_layout(width=300, height=180, margin=dict(l=10, r=30, t=30, b=10), autosize=False)
        st.plotly_chart(fig_gauge)

# Footer
st.markdown("<style>.stApp > footer {display: none;}</style>", unsafe_allow_html=True)  # Hide footer to save space
