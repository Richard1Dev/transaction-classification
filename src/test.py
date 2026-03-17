# test.py
import pickle
import pandas as pd
from utils import load_and_split, evaluate_model

def run_test():
    """
    Loads the serialised champion model and performs a final evaluation
    on the held-out test set to verify performance and persistence.
    """
    print("Testing the champion model...")

    # 1. Load the test data (stratified split ensures consistency with training)
    _, X_test, _, y_test = load_and_split('../data/creditcard.csv')

    # 2. Load the Pipeline
    try:
        with open('../models/champion_fraud_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model file not found. Ensure train.py has been run.")
        return

    # 3. Generate Predictions
    # The pipeline handles scaling internally, so raw X_test is passed directly.
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models like LinearSVC that use decision_function
        y_scores = pipeline.decision_function(X_test)

    # 4. Final Evaluation
    # I have added the filename parameter to ensure visual proof is saved to /figures
    evaluate_model(
        y_test, 
        y_scores, 
        model_name="Champion_Final_Test", 
        filename="final_verification_pr_curve.png"
    )

if __name__ == "__main__":
    run_test()
