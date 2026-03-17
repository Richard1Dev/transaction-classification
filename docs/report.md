# Technical Report: Credit Card Fraud Detection System

## 1. Executive Summary
This project successfully developed a machine learning system to identify fraudulent credit card transactions. Given the extreme rarity of fraud (0.17% of total volume), I engineered the system to prioritise detection (Recall) and the accuracy of alarms (Precision) over standard accuracy. The final champion model, an **XGBoost Classifier**, achieved an **AUPRC of 0.88**, providing a robust balance between fraud capture and customer friction.

## 2. Exploratory Data Analysis & Insights
I conducted a detailed analysis to understand the underlying patterns of fraudulent activity:
* **Feature Importance:** Latent features derived from PCA (V17, V14, and V12) showed the strongest correlation with fraudulent behaviour. Visual separation in these features is documented in the `figures/` directory.
* **Transaction Amount:** Fraudulent transactions often exhibit skewed distributions, including $0.00 "ping" attempts, necessitating robust normalisation.
* **Temporal Patterns:** Fraudulent activity does not strictly follow the diurnal cycles of genuine users, appearing more consistently across a 24-hour period.
* **Class Imbalance:** Accuracy is misleading here. A trivial baseline model that predicts every transaction as genuine would achieve about 99.83% accuracy, while completely failing to detect any fraudulent cases.

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

To further enhance the robustness, interpretability, and real-world applicability of the fraud detection system, the following improvements are recommended:

1. **Threshold Optimisation:**
   Replace the default 0.5 decision threshold with an optimised value based on business objectives. This can be achieved by maximising F1 score or enforcing a minimum precision/recall constraint using the Precision-Recall curve. Reporting performance at this optimal threshold will provide a more realistic view of operational effectiveness.

2. **Advanced Evaluation Metrics:**
   Extend evaluation beyond AUPRC by incorporating:

   * Confusion matrices for clear visibility into false positives and false negatives
   * Precision@K to simulate real-world fraud investigation queues (e.g. top 100 flagged transactions)
   * Baseline comparison using the fraud rate to contextualise PR-AUC performance

3. **Hyperparameter Optimisation:**
   Integrate `GridSearchCV`, `RandomizedSearchCV`, or a framework such as Optuna within the pipeline to systematically tune model parameters (e.g. `learning_rate`, `max_depth`, `n_estimators`) while preserving the no-leakage design.

4. **Model Calibration:**
   Apply probability calibration techniques (e.g. Platt scaling or isotonic regression via `CalibratedClassifierCV`) to ensure predicted probabilities reflect true likelihoods, which is critical for threshold-based decision systems.

5. **Explainability and Model Transparency:**
   Introduce model interpretability using:

   * Feature importance from the final ensemble model
   * SHAP (SHapley Additive exPlanations) for both global feature influence and local transaction-level explanations
     This is particularly important in financial systems requiring auditability.

6. **Temporal Validation Strategy:**
   Replace random train-test splitting with a time-based split to better simulate real-world deployment conditions, where models are trained on past data and evaluated on future transactions.

7. **Concept Drift and Data Monitoring:**
   Implement monitoring mechanisms to detect distributional shifts over time, such as:

   * Population Stability Index (PSI)
   * Feature distribution tracking
     This ensures continued model reliability as fraud patterns evolve.

8. **Cost-Sensitive Evaluation:**
   Incorporate a business-driven cost framework that assigns explicit penalties to false negatives (missed fraud) and false positives (customer friction). This allows optimisation of the model based on financial impact rather than abstract metrics alone.

9. **Anomaly Detection Layer:**
   Augment the supervised model with unsupervised techniques such as Isolation Forest or One-Class SVM to identify previously unseen ("zero-day") fraud patterns that deviate from historical behaviour.

10. **Deployment and Inference Layer:**
    Package the trained pipeline into a production-ready service using FastAPI or Docker. The inference layer should:

    * Accept raw transaction inputs
    * Apply the full preprocessing pipeline
    * Output calibrated fraud probabilities with a configurable decision threshold

11. **Experiment Tracking and Configuration Management:**
    Introduce structured experiment tracking (e.g. logging metrics, parameters, and model versions) and externalise model configurations into YAML/JSON files to improve reproducibility and maintainability.

Collectively, these enhancements transition the project from a strong experimental pipeline to a production-grade fraud detection system aligned with real-world financial risk management requirements.
