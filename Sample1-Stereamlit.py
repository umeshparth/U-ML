import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Simulate dataset (replace with your actual dataset if available)
data = pd.DataFrame({
    'education': np.random.choice(['Graduate', 'Not Graduate'], 1000),
    'self_employed': np.random.choice(['Yes', 'No'], 1000),
    'income_annum': np.random.normal(500000, 150000, 1000),
    'loan_amount': np.random.normal(200000, 75000, 1000),
    'loan_term': np.random.randint(1, 20, 1000),
    'cibil_score': np.random.randint(300, 900, 1000),
    'residential_assets_value': np.random.normal(300000, 100000, 1000),
    'commercial_assets_value': np.random.normal(150000, 75000, 1000),
    'luxury_assets_value': np.random.normal(200000, 80000, 1000),
    'bank_asset_value': np.random.normal(100000, 50000, 1000),
    'loan_status': np.random.choice([0, 1], 1000)
})

# If you have your own dataset, uncomment and modify this line:
# data = pd.read_csv('path_to_your_dataset.csv')

# Preprocessing
def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    le = LabelEncoder()
    for col in ['education', 'self_employed']:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    scaler = StandardScaler()
    numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                      'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler

X, y, scaler = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with hyperparameter tuning
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

#TODO with Decision Tree

# Evaluate
y_pred = best_rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, y_pred))

# # Save model and scaler
# with open('best_model.pkl', 'wb') as f:
#     pickle.dump(best_rf, f)
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# print("Model and scaler saved as 'best_model.pkl' and 'scaler.pkl'.") 

# # Load the model and scaler
# with open('pkl/best_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('pkl/scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# Streamlit app
st.title("Loan Approval Prediction")
st.write("Enter the applicant details to predict loan approval status.")

# Input fields
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
income_annum = st.number_input("Annual Income (in INR)", min_value=0.0, value=500000.0)
loan_amount = st.number_input("Loan Amount (in INR)", min_value=0.0, value=200000.0)
loan_term = st.number_input("Loan Term (in years)", min_value=1, max_value=30, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=0.0, value=300000.0)
commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=0.0, value=150000.0)
luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=0.0, value=200000.0)
bank_asset_value = st.number_input("Bank Asset Value (in INR)", min_value=0.0, value=100000.0)

# Data CLeaning
# Data TRaining 

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'education': [1 if education == 'Graduate' else 0],
        'self_employed': [1 if self_employed == 'Yes' else 0],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })
    
    # Scale numerical features
    numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                      'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Make prediction
    #TODO
    prediction = best_rf.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Not Approved"
    
    # Display result
    st.success(f"Loan Status: {result}")

# Add some styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
