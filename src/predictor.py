import pickle
import pandas as pd

class FraudDetector:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)  # this is the full Pipeline

    def predict_proba(self, raw_df: pd.DataFrame):
        """
        Returns fraud probability for each row.
        The pipeline handles scaling internally.
        """
        return self.pipeline.predict_proba(raw_df)[:, 1]

    def predict(self, raw_df: pd.DataFrame, threshold=0.5):
        """
        Returns binary predictions using a configurable threshold.
        """
        probs = self.predict_proba(raw_df)
        return (probs >= threshold).astype(int)
