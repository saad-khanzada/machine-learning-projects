# Titanic Survival Classification

An exploratory machine-learning notebook that analyzes Titanic passenger data and builds binary-classification models to predict survival.

## Project Workflow

The notebook covers:

1. Loading the Titanic passenger dataset
2. Exploratory data analysis and visualization
3. Missing-value inspection and treatment
4. Feature removal and categorical encoding
5. Stratified training and test split
6. Hyperparameter tuning with `GridSearchCV`
7. Comparison of SVM, Decision Tree, and Random Forest classifiers
8. Accuracy, precision, recall, F1-score, classification reports, and confusion matrices

## Verified Results

The saved notebook outputs report the following held-out test results on 178 passengers:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| SVM | 76.97% | 76.66% | 76.97% | 76.60% |
| Decision Tree | 78.65% | 79.63% | 78.65% | 77.41% |
| **Random Forest** | **79.78%** | **79.61%** | **79.78%** | **79.65%** |

Best Random Forest parameters found by `GridSearchCV`:

```python
{
    "criterion": "gini",
    "max_depth": 10,
    "n_estimators": 200
}
```

## Repository Files

```text
Titanic-Classification/
├── README.md
├── requirements.txt
└── titanic_classification_project.ipynb
```

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

## Running the Notebook

1. Download or clone the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain a Titanic dataset with the columns used by the notebook.
4. In Google Colab, upload it as `/content/titanic_dataset.csv`, or update the path in the data-loading cell.
5. Remove the misspelled `import skleaSSrn` line in the first cell before running the notebook. The required scikit-learn components are imported later using `sklearn`.
6. Run the cells sequentially.

## Current Reproducibility Status

The uploaded notebook is preserved as submitted, including its saved outputs. The performance table above is taken directly from those outputs; reproducing it requires the same dataset, preprocessing, library versions, and random state.

Known setup items:

- The notebook expects `/content/titanic_dataset.csv`.
- The first cell contains a misspelled `skleaSSrn` import that should be removed or corrected.
- Some visualization functions may show deprecation warnings with newer Seaborn versions.

## Suggested Next Improvements

- Add a legally redistributable dataset source or download instructions.
- Correct the import typo and use a portable dataset path.
- Add a clear project introduction and learning objectives.
- Add a confusion-matrix visualization for each final model.
- Record the final dependency versions after successful execution.

## Disclaimer

This project is intended for education and portfolio demonstration. Predictions are based on a historical teaching dataset and should not be interpreted beyond that context.
