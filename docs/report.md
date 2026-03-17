# Technical Report: Credit Card Fraud Detection System

## 1. Executive Summary
This project successfully developed a machine learning system to identify fraudulent credit card transactions. Given the extreme rarity of fraud (0.17% of total volume), I engineered the system to prioritise detection (Recall) and the accuracy of alarms (Precision) over standard accuracy. The final champion model, an **XGBoost Classifier**, achieved an **AUPRC of 0.88**, providing a robust balance between fraud capture and customer friction.

## 2. Exploratory Data Analysis & Insights
I conducted a detailed analysis to understand the underlying patterns of fraudulent activity:
* **Feature Importance:** Latent features derived from PCA (V17, V14, and V12) showed the strongest correlation with fraudulent behaviour. Visual separation in these features is documented in the `figures/` directory.
* **Transaction Amount:** Fraudulent transactions often exhibit skewed distributions, including $0.00 "ping" attempts, necessitating robust normalisation.
* **Temporal Patterns:** Fraudulent activity does not strictly follow the diurnal cycles of genuine users, appearing more consistently across a 24-hour period.

## 3. Data Engineering & Pipeline Design
To ensure the model generalises to real-world data without bias, I implemented a strict "No-Leakage" architecture:
* **Robust Scaling:** I utilised the `RobustScaler` to handle financial outliers that would otherwise distort standard scaling methods.
* **Stratified Sampling:** All data splits maintained the exact 0.17% fraud ratio to ensure the test set remained a realistic representation of the population.
* **Encapsulation:** By using `imblearn.pipeline`, I ensured that all preprocessing steps (scaling and model fitting) were contained strictly within training folds, preventing look-ahead bias.

## 4. Modelling Experiments & Key Insights
I benchmarked several algorithms to identify the optimal balance of performance and reliability.

| Model | AUPRC | Complexity | Status |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Basic) | ~0.75 | Low | Baseline |
| Logistic Regression (Weighted) | ~0.76 | Low | Improved Baseline |
| Random Forest | ~0.85 | Medium | Candidate |
| **XGBoost** | **0.88** | **High** | **Champion** |

**The Class Weighting Insight:**
During experimentation, I observed that a "Naive" model with data leakage achieved ~0.75 AUPRC. Interestingly, an "Honest" model (no leakage) using `class_weight='balanced'` achieved ~0.76. This confirms that for imbalanced fraud data, properly calibrating the model to the 0.17% minority class is more impactful than the artificial gains provided by improper scaling.

## 5. Performance Metrics
I avoided the "Accuracy Trap." A model that predicts 100% genuine would achieve 99.83% accuracy but 0.00% detection. Instead, I utilised:
* **AUPRC (Area Under the Precision-Recall Curve):** My primary metric, focusing on the quality of the minority class detection.
* **F1 Score:** Evaluated at various thresholds to determine the best point for operational deployment.

## 6. Operational Recommendations
The model is serialised and ready for integration. I recommend:
1.  **Threshold Tuning:** Depending on the business cost of a False Positive (annoying a customer) vs. a False Negative (allowing fraud), the probability threshold should be adjusted.
2.  **Monitoring:** Monthly re-evaluation of the AUPRC is advised to detect "Concept Drift" as fraudsters change their tactics.

## 7. Future Improvements
1.  **Hyperparameter Tuning:** Implementing `GridSearchCV` or `Optuna` within the pipeline to fine-tune the XGBoost parameters (e.g. `learning_rate` and `max_depth`).
2.  **Anomaly Detection:** Layering in an isolation forest or one-class SVM specifically for "zero-day" fraud types that do not resemble historical examples.
3.  **Deployment:** Packaging the `src/` folder into a Docker container or a FastAPI endpoint for real-time transaction scoring.
