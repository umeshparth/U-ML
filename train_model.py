import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

# Simulate dataset
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Evaluate
y_pred = best_rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved.")