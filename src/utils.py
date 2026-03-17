# utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def evaluate_model(y_test, y_scores, model_name="Model", filename=None):
    """
    Comprehensive evaluation for imbalanced fraud data.
    
    Now prints Precision and Recall at the default 0.5 threshold 
    in addition to AUPRC and F1.
    """
    # 1. Calculate values for the PR Curve (AUPRC)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
    auprc = auc(recall_curve, precision_curve)
    
    # 2. Calculate point-in-time metrics at 0.5 threshold
    y_pred = [1 if x >= 0.5 else 0 for x in y_scores]
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    # 3. Print Results
    print(f"\n--- {model_name} Results ---")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f} (Accuracy of Alarms)")
    print(f"Recall:    {rec:.4f} (Detection Rate)")
    
    # 4. Plotting Logic
    plt.figure(figsize=(8, 5))
    plt.plot(recall_curve, precision_curve, label=f'{model_name} AUC={auprc:.4f}')
    plt.fill_between(recall_curve, precision_curve, alpha=0.1)
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.xlabel('Recall (Detection Rate)')
    plt.ylabel('Precision (Accuracy of Alarms)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if filename:
        os.makedirs('../figures', exist_ok=True)
        save_path = os.path.join('../figures', filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
    plt.show()
    plt.close()
    
    return auprc

def get_fraud_ratio(y):
    """Calculates the ratio of fraud cases to total cases."""
    return (y == 1).sum() / len(y)

def load_and_split(path, test_size=0.2):
    """
    Loads the credit card data and performs a stratified split.
    Stratification is utilised to maintain the 0.17% fraud ratio in both sets.
    """
    data = pd.read_csv(path)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
