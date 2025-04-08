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
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            'XGBoost': XGBClassifier(random_state=42, max_depth=6, learning_rate=0.1, n_estimators=100)
        }
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_score = 0

    def load_or_simulate_data(self):
        """Load or simulate dataset for training."""
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
            'loan_status': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Imbalanced classes
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
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print("Classification Report:\n", classification_report(self.y_test, y_pred))

            # Track the best model based on ROC-AUC
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model = model

    def save_best_model(self):
        """Save the best model and scaler to files."""
        if self.best_model is None or self.scaler is None:
            raise ValueError("No best model or scaler available. Train models first.")

        if not os.path.exists('models_Test'):
            os.makedirs('models_test')

        with open('models_test/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        with open('models_test/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\nBest model ({type(self.best_model).__name__}) and scaler saved to 'models/' directory.")

    def run(self):
        """Execute the training pipeline."""
        # Load or simulate data
        data = self.load_or_simulate_data()

        # Preprocess data
        X, y = self.preprocess_data(data)

        # Train and evaluate models
        self.train_and_evaluate_models(X, y)

        # Save the best model
        self.save_best_model()

if __name__ == "__main__":
    # Create and run the trainer
    trainer = TrainModel()
    trainer.run()