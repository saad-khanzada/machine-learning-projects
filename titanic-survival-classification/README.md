# Titanic Survival Classification

A machine learning classification project that explores Titanic passenger data, prepares the dataset for modeling, tunes three classification algorithms, and compares their performance on a held-out test set.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saad-khanzada/machine-learning-projects/blob/main/titanic-survival-classification/titanic_classification_project.ipynb)

## Project Objective

The notebook uses passenger information to predict the binary target `Survived`. It covers exploratory data analysis, missing-data handling, categorical encoding, train/test splitting, hyperparameter tuning, model evaluation, and metric-based model comparison.

## Dataset Snapshot

The dataset loaded by the notebook contains **891 rows and 12 columns**.

| Item | Verified notebook value |
|---|---|
| Rows | 891 |
| Original columns | 12 |
| Target | `Survived` |
| Missing `Age` values | 177 |
| Missing `Cabin` values | 687 |
| Missing `Embarked` values | 2 |
| Rows after missing-data handling | 889 |
| Final model features | 8 |
| Training rows | 711 |
| Test rows | 178 |

The original columns shown in the notebook are:

`PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

> The dataset file itself is not stored in this project folder. The notebook currently expects it at `/content/titanic_dataset.csv`.

## Exploratory Data Analysis

The notebook includes saved visual outputs for the following analyses:

- Distribution of `Parch`
- Age distribution
- Age versus fare, separated by passenger class and sex
- Survival rate by sex and passenger class
- Age distribution by sex
- Age versus survival status
- Survival by sex and passenger class using a point plot
- Missing-data heatmap
- Age by passenger class using a box plot

The notebook's written observations include that many passengers fall roughly in the 20–40 age range and that the plotted survival rates are higher for women than men in the displayed comparisons.

## Data Cleaning and Preprocessing

### 1. Age imputation

Missing `Age` values are filled using fixed values based on `Pclass`:

| Passenger class | Imputed age |
|---|---:|
| 1 | 37 |
| 2 | 29 |
| 3 | 24 |

### 2. Column removal

The notebook removes:

- `Cabin` because it contains many missing values
- `Name`
- `Ticket`
- `PassengerId`

### 3. Remaining missing values

After age imputation and removing `Cabin`, the notebook uses `dropna()`. The resulting dataset contains **889 rows**, which removes the two rows with missing `Embarked` values.

### 4. Categorical encoding

`Sex` and `Embarked` are converted to categorical data and then encoded with `pd.get_dummies(..., drop_first=True)`.

The resulting indicator columns are:

- `male`
- `Q`
- `S`

After encoding, the model-ready dataframe contains **9 columns total**: the target plus **8 input features**.

### Final model features

```text
Pclass
Age
SibSp
Parch
Fare
male
Q
S
```

No explicit feature-scaling step is present in the notebook.

## Train/Test Split

The notebook uses:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Verified split sizes:

- **Training set:** 711 rows
- **Test set:** 178 rows
- **Test class support:** 110 for class 0 and 68 for class 1

## Models Trained

Three classifiers are actually included in the training and tuning loop:

1. Support Vector Machine (SVM)
2. Decision Tree
3. Random Forest

Although `XGBClassifier` is imported in the notebook, XGBoost is **not** included in the `models` dictionary and is not trained or evaluated in the saved workflow.

## Hyperparameter Tuning

The notebook uses `GridSearchCV` with:

- **5-fold `KFold` cross-validation**
- `shuffle=True`
- `random_state=42`
- Accuracy as the scoring metric
- `n_jobs=-1`

### SVM search space

```python
{
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10]
}
```

**6 parameter combinations**, producing **30 fits** across 5 folds.

Best saved parameters:

```python
{"C": 0.1, "kernel": "linear"}
```

### Decision Tree search space

```python
{
    "max_depth": [3, 5, 10, None],
    "criterion": ["gini", "entropy"]
}
```

**8 parameter combinations**, producing **40 fits** across 5 folds.

Best saved parameters:

```python
{"criterion": "entropy", "max_depth": 5}
```

### Random Forest search space

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "criterion": ["gini", "entropy"]
}
```

**18 parameter combinations**, producing **90 fits** across 5 folds.

Best saved parameters:

```python
{
    "criterion": "gini",
    "max_depth": 10,
    "n_estimators": 200
}
```

## Verified Test Results

The following values come directly from the saved notebook outputs.

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| SVM | 76.97% | 76.66% | 76.97% | 76.60% |
| Decision Tree | 78.65% | 79.63% | 78.65% | 77.41% |
| **Random Forest** | **79.78%** | **79.61%** | **79.78%** | **79.65%** |

Based on the saved test accuracy, **Random Forest has the highest value among the three evaluated models**.

### Random Forest class-level report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.82 | 0.85 | 0.84 | 110 |
| 1 | 0.75 | 0.71 | 0.73 | 68 |

The notebook also generates separate bar charts comparing all three models on:

- Accuracy
- Precision
- Recall
- F1-score

## Evaluation Functionality

For each tuned model, the notebook:

1. Fits `GridSearchCV` on the training set
2. Retrieves the best estimator
3. Predicts the held-out test set
4. Calculates accuracy
5. Generates a classification report
6. Calculates a confusion matrix
7. Stores weighted precision, recall, and F1-score
8. Adds the results to a comparison dataframe

Confusion matrices are calculated and stored in the `confusion_matrices` dictionary, but the notebook does not display or plot those matrices in its saved outputs.

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

The project `requirements.txt` also lists `xgboost` and `jupyter`. XGBoost is imported by the notebook but is not one of the three trained models.

## Project Files

```text
titanic-survival-classification/
├── README.md
├── requirements.txt
└── titanic_classification_project.ipynb
```

## Run in Google Colab

Use the **Open in Colab** badge at the top of this README.

The notebook currently loads the dataset from:

```text
/content/titanic_dataset.csv
```

To run it in Colab:

1. Open the notebook using the badge above.
2. Upload the expected Titanic CSV file as `titanic_dataset.csv`.
3. Make sure it is available at `/content/titanic_dataset.csv`.
4. Correct or remove the misspelled `import skleaSSrn` line in the first code cell.
5. Run the notebook cells in order.

## Run Locally

Clone the repository and install the listed dependencies:

```bash
pip install -r requirements.txt
```

Then update the dataset path in the notebook from `/content/titanic_dataset.csv` to the local location of your CSV file.

## Current Reproducibility Notes

The README intentionally documents the project as it currently exists rather than claiming a cleaner workflow than the notebook actually contains.

- The dataset is not included in this project folder, and its original source is not documented in the notebook.
- The first code cell contains the invalid import `import skleaSSrn`, so a fresh run requires correcting or removing that line.
- `RandomForestClassifier()` and `DecisionTreeClassifier()` are created without an explicit `random_state`; exact tuning results may therefore vary between reruns.
- The notebook suppresses warnings globally.
- `sns.distplot` is used for the age distribution and may produce deprecation-related behavior with newer Seaborn versions.
- `XGBClassifier` and `cross_val_score` are imported but are not used in the actual training workflow.
- The final reporting cell contains a general sentence mentioning XGBoost, but XGBoost is not trained or evaluated in this notebook.
- Confusion matrices are computed but not displayed.
- The saved model metrics above reflect the outputs currently stored in the notebook.

## Scope

This is an educational machine learning project demonstrating a complete classification workflow from exploratory analysis and preprocessing through model tuning and evaluation. The reported metrics describe this notebook's saved test split and should not be interpreted as performance beyond this dataset and workflow.
