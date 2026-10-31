# AI300 Projects — Customer Churn Prediction

Stage 1 research for the AI300 capstone project: exploring TrustTelecom customer data and selecting a classifier to identify customers at risk of churn.

## Research notebook

Open [AI300 Capstone - Stage 1.ipynb](AI300%20Capstone%20-%20Stage%201.ipynb) to review the step-by-step workflow and saved outputs.

The notebook covers:

1. Read-only SQL retrieval from the remote `ai300_capstone` MySQL database.
2. Live schema verification, join-key checks and customer-level table joins.
3. Data cleaning, missing-label handling and feature engineering.
4. Training-set exploratory analysis with charts and group summaries.
5. A dummy baseline and hyperparameter tuning for Logistic Regression and Random Forest.
6. Selection by five-fold cross-validation ROC AUC, followed by evaluation of the selected model on a reserved test set.

Preprocessing is fitted inside each cross-validation fold. Customer status, churn reasons and identifiers are excluded from model inputs. Refunds and quarterly totals are also excluded until their availability before churn can be confirmed.

## Results

The saved execution retrieved 7,043 customer records on 30 October 2026. It excluded 52 missing target labels, leaving 6,991 labeled customers: 5,592 for training and 1,399 for testing. The random seed is 42.

| Model | Best mean CV ROC AUC |
| --- | ---: |
| Logistic Regression | 0.8971 |
| Random Forest | **0.9013** |

The selected Random Forest achieved **0.9068 test ROC AUC** and **0.7876 test average precision**. Its approximate 95% bootstrap interval for test ROC AUC is 0.8904–0.9233.

Selected configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=None,
    max_features=0.6,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=1,
)
```

These are retrospective research results. The snapshot has no feature timestamps proving that every predictor preceded churn. Confirm feature timing and evaluate on a later snapshot before operational use. The model family was selected using cross-validation, before inspecting its test results.

## Run locally

Use Python 3.11 or later with Jupyter Notebook, JupyterLab or VS Code's Jupyter extension.

```bash
python -m pip install pandas numpy matplotlib seaborn sqlalchemy pymysql scikit-learn ipykernel notebook
python -m notebook
```

Open the notebook and run its cells from top to bottom. The saved run used Python 3.15.0, pandas 3.1.0 and scikit-learn 1.9.1; results may vary with package versions or database changes.

## Repository scope

- `AI300 Capstone - Stage 1.ipynb`: research notebook with executed charts and results.

Stage 1 is complete. Model serialization, Flask development and cloud deployment are later milestones and are not included here.
