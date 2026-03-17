# train.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from utils import evaluate_model, load_and_split

def run_experiment():
    print("--- Initialising Production Training Pipeline ---")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_split('../data/creditcard.csv')
    print(f"Training on {len(X_train):,} samples with {y_train.sum()} fraud cases.")

    # 2. Setup Cross-Validation
    # StratifiedKFold is essential to maintain the 0.17% fraud ratio in every fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define weight for class imbalance
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # 3. Define Model Suite
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(class_weight='balanced'),
        "XGBoost": XGBClassifier(scale_pos_weight=ratio, eval_metric='logloss'),
        "Linear_SVM": LinearSVC(class_weight='balanced', max_iter=5000, dual=False)
    }

    best_cv_auprc = 0.0
    best_pipe = None
    best_name = "None"

    # 4. Iterative Benchmarking with CV
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Create Pipeline: Scaling must happen inside CV to prevent leakage
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', model)
        ])
        
        # Perform Cross-Validation on the training set
        # We use 'average_precision' as the scoring metric for AUPRC
        cv_scores = cross_val_score(
            pipe, X_train, y_train, 
            cv=skf, 
            scoring='average_precision', 
            n_jobs=-1
        )
        
        mean_auprc = np.mean(cv_scores)
        std_auprc = np.std(cv_scores)
        
        print(f"  > Mean CV AUPRC: {mean_auprc:.4f} (+/- {std_auprc:.4f})")

        # 5. Final Fit & Test Set Vibe Check
        # After CV, we fit on the WHOLE training set to test on the unseen hold-out
        pipe.fit(X_train, y_train)
        
        if hasattr(pipe, "predict_proba"):
            y_scores = pipe.predict_proba(X_test)[:, 1]
        else:
            y_scores = pipe.decision_function(X_test)

        test_auprc = evaluate_model(y_test, y_scores, model_name=f"{name}_Holdout")

        # Track the Champion based on CV performance (more reliable than single split)
        if mean_auprc > best_cv_auprc:
            best_cv_auprc = mean_auprc
            best_pipe = pipe
            best_name = name

    # 6. Serialise the Champion
    if best_pipe:
        os.makedirs('../models', exist_ok=True)
        model_path = '../models/champion_fraud_model.pkl'
        
        print(f"\n--- CHAMPION IDENTIFIED: {best_name} ---")
        print(f"Saving model with CV AUPRC: {best_cv_auprc:.4f}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_pipe, f)
        print(f"Serialisation complete: {model_path}")
    else:
        print("\nError: No model was successfully validated.")

if __name__ == "__main__":
    run_experiment()
