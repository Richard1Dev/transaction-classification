import pickle
import pandas as pd

class FraudDetector:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]

    def predict_safe(self, raw_df):
        """Standardises and predicts in one go."""
        df = raw_df.copy()
        
        # Apply the scaling we learned during training
        df['Amount'] = self.scaler.transform(df[['Amount']])
        df['Time'] = self.scaler.transform(df[['Time']])
        
        # Ensure the model gets exactly what it expects
        return self.model.predict_proba(df[self.features])[:, 1]
