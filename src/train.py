# train.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from utils import evaluate_model, load_and_split

def run_experiment():
    print("Loading data...")
    # Path assumes execution from the src/ directory or via %run in notebooks
    X_train, X_test, y_train, y_test = load_and_split('../data/creditcard.csv')

    # Define weight for class imbalance
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Model suite with cost-sensitive parameters enabled
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(class_weight='balanced'),
        "XGBoost": XGBClassifier(scale_pos_weight=ratio),
        "Linear_SVM": LinearSVC(class_weight='balanced', max_iter=5000, dual=False)
    }

    # Initialise variables to avoid UnboundLocalError and track the champion
    best_auprc = 0.0
    best_pipe = None
    best_name = "None"

    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Build the pipeline to ensure scaling is fitted strictly on training data
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', model)
        ])
        
        pipe.fit(X_train, y_train)
        
        # Extract scores based on model capability (probabilities vs decision boundaries)
        if hasattr(pipe, "predict_proba"):
            y_scores = pipe.predict_proba(X_test)[:, 1]
        else:
            y_scores = pipe.decision_function(X_test)

        # Standard evaluation; plots are suppressed here to avoid console clutter
        current_auprc = evaluate_model(y_test, y_scores, model_name=name)

        if current_auprc > best_auprc:
            best_auprc = current_auprc
            best_pipe = pipe
            best_name = name

    # Serialise the Champion Model for production use
    if best_pipe:
        print(f"\nSaving best model ({best_name}) with AUPRC: {best_auprc:.4f}")
        with open('../models/champion_fraud_model.pkl', 'wb') as f:
            pickle.dump(best_pipe, f)
    else:
        print("\nNo model was successfully trained or saved.")

if __name__ == "__main__":
    run_experiment()
