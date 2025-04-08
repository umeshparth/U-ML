import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import os

class TrainModel:
    def __init__(self):
        """Initialize the TrainModel class."""
        self.models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=15),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15),
            'XGBoost': XGBClassifier(random_state=42, max_depth=8, learning_rate=0.05, n_estimators=200)
        }
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_score = 0

    def load_or_simulate_data(self):
        """Load or simulate dataset for training with meaningful patterns across all features."""
        np.random.seed(42)
        n_samples = 1000
        cibil_score = np.random.randint(300, 900, n_samples)
        income_annum = np.random.normal(500000, 150000, n_samples)
        loan_amount = np.random.normal(200000, 75000, n_samples)
        loan_term = np.random.randint(1, 20, n_samples)
        residential_assets_value = np.random.normal(300000, 100000, n_samples)
        commercial_assets_value = np.random.normal(150000, 75000, n_samples)
        luxury_assets_value = np.random.normal(200000, 80000, n_samples)
        bank_asset_value = np.random.normal(100000, 50000, n_samples)
        education = np.random.choice(['Graduate', 'Not Graduate'], n_samples)
        self_employed = np.random.choice(['Yes', 'No'], n_samples)

        # Calculate total assets
        total_assets = (residential_assets_value + commercial_assets_value +
                       luxury_assets_value + bank_asset_value)

        # Decision rule for loan_status
        # Weighted score: Higher cibil_score, income, assets, education (Graduate), and lower loan_amount, loan_term favor approval
        score = (0.3 * (cibil_score / 900) +  # Normalize cibil_score (0-1)
                 0.2 * (income_annum / 1000000) +  # Normalize income (0-1)
                 0.2 * (total_assets / 1000000) +  # Normalize total assets (0-1)
                 0.1 * np.where(education == 'Graduate', 1, 0) +  # Education boost
                 0.1 * np.where(self_employed == 'No', 1, 0) +  # Stable employment boost
                 0.1 * (1 - (loan_amount / 500000)) +  # Penalize high loan amount (0-1)
                 0.1 * (1 - (loan_term / 20)))  # Penalize long term (0-1)

        # Normalize score to 0-1
        score = np.clip(score, 0, 1)
        # Add noise and threshold for loan_status
        noise = np.random.normal(0, 0.1, n_samples)  # Small noise
        loan_status = np.where(score + noise > 0.7, 1, 0)  # Threshold at 0.7

        data = pd.DataFrame({
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value,
            'loan_status': loan_status
        })
        return data

    def preprocess_data(self, df):
        """Preprocess the dataset including encoding and scaling."""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        for col in ['education', 'self_employed']:
            df[col] = df[col].map({'Graduate': 1, 'Not Graduate': 0, 'Yes': 1, 'No': 0})

        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        self.scaler = StandardScaler()
        numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                         'residential_assets_value', 'commercial_assets_value',
                         'luxury_assets_value', 'bank_asset_value']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        return X, y

    def train_and_evaluate_models(self, X, y):
        """Train and evaluate multiple models."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print("Classification Report:\n", classification_report(self.y_test, y_pred))

            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model = model

    def save_best_model(self):
        """Save the best model and scaler to files."""
        if self.best_model is None or self.scaler is None:
            raise ValueError("No best model or scaler available. Train models first.")

        if not os.path.exists('models'):
            os.makedirs('models')

        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\nBest model ({type(self.best_model).__name__}) and scaler saved to 'models/' directory.")

    def run(self):
        """Execute the training pipeline."""
        data = self.load_or_simulate_data()
        X, y = self.preprocess_data(data)
        self.train_and_evaluate_models(X, y)
        self.save_best_model()

if __name__ == "__main__":
    trainer = TrainModel()
    trainer.run()