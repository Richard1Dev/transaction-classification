# Credit Card Fraud Detection

A machine learning pipeline designed to identify fraudulent transactions with high precision and recall. This project focuses on handling extreme class imbalance and preventing data leakage through a robust pipeline architecture.

## Project Structure

The project is organised as follows:

```
credit-card-fraud-ml/
│
├── data/           # Source CSV data (e.g. creditcard.csv)
├── docs/           # Technical report and documentation
├── figures/        # Exported Precision-Recall curves and EDA plots
├── models/         # Serialised champion model (.pkl)
├── notebooks/      # EDA and benchmarking experiments
├── src/            # Python scripts and utilities
├── requirements.txt
└── README.md
```

## Methodology

* **Anti-Leakage Pipeline:** Utilises `imblearn.pipeline.Pipeline` to ensure that the `RobustScaler` is fitted strictly on training data, eliminating look-ahead bias.
* **Cost-Sensitive Learning:** Addresses the extreme 0.17% fraud imbalance by utilising `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Linear models. This proved more effective than naive scaling during experimentation.
* **Metric Focus:** Prioritises **AUPRC** (Area Under the Precision-Recall Curve) to ensure high detection (Recall) while minimising false alarms (Precision).
* **Model Suite:** Compares Logistic Regression, Random Forest, Hist-Gradient Boosting, XGBoost, and Linear SVM.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Explore the Research:**
    Review `notebooks/eda.ipynb` for data insights and `notebooks/experiments.ipynb` to see the benchmarking process and leakage analysis.

3.  **Train the Champion Model:**
    Execute the training suite to benchmark all models and serialise the best performer to the `models/` directory:
    ```bash
    python src/train.py
    ```

4.  **Verify Results:**
    Run the final verification script to load the serialised pipeline and test it against the held-out dataset. This will also save a final verification plot to `figures/`:
    ```bash
    python src/test.py
    ```
