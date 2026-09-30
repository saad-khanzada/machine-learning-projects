# Titanic Survival Classification

Predict passenger survival from demographic and travel information using an end-to-end scikit-learn workflow.

**Author: Saad Kabeer**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saad-khanzada/machine-learning-projects/blob/main/titanic-survival-classification/titanic_classification_project.ipynb)

[View notebook](titanic_classification_project.ipynb) · [Dataset](titanic_dataset.csv) · [Requirements](requirements.txt)

## Project overview

The notebook validates the data, explores the training partition, compares simple and engineered features, and evaluates multiple classification pipelines. Model selection uses training cross-validation; the reserved test set is used for final evaluation.

| Dataset detail | Value |
|---|---|
| Passengers / original columns | 891 / 12 |
| Target | `Survived`: 0 = did not survive, 1 = survived |
| Training / test rows | 712 / 179 |
| Split | Stratified 80/20, random seed 101 |
| Missing values | Age: 177; Cabin: 687; Embarked: 2 |

All 891 passengers are retained. The exact CSV used in the saved run is included beside the notebook. Its original download source has not been documented; no source attribution is assumed.

## Modeling workflow

1. **Validate and split:** Check columns, labels, passenger IDs, and numeric values; reserve the test set before relationship analysis.
2. **Explore training data:** Plot age and fare distributions, age by class, survival rates, correlations, and missing values.
3. **Construct features:** Compare seven simple predictors with an extended set containing family size, solo-traveler status, fare per person, name title, and cabin availability.
4. **Preprocess within pipelines:** Fit median and most-frequent imputers inside each training fold, one-hot encode categories, and scale numeric features for Logistic Regression.
5. **Compare models:** Evaluate Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting on the same five stratified folds. Include a majority-class dummy baseline.
6. **Tune and select:** Run `RandomizedSearchCV` on the two highest-scoring initial non-dummy candidates, with 16 parameter combinations each. Select by mean CV accuracy.
7. **Evaluate and inspect errors:** Report test metrics, a confusion matrix, ROC curve, and misclassified passengers. Demonstrate prediction from raw passenger rows.

The final model is selected before viewing test scores. A higher test score from a comparator does not change that selection.

## Recorded results

The saved run selected **tuned Gradient Boosting with simple features**, with **83.70% mean CV accuracy** (fold standard deviation: 2.96 percentage points).

| Evaluated model | Test accuracy | Survivor precision | Survivor recall | Survivor F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CV-selected tuned Gradient Boosting | 81.01% | 87.23% | 59.42% | 70.69% | 82.63% |
| Initial Gradient Boosting, simple features | 82.68% | 86.54% | 65.22% | 74.38% | 83.37% |
| Majority-class baseline | 61.45% | 0.00% | 0.00% | 0.00% | 50.00% |

The selected model correctly classified **145 of 179** test passengers. Its survivor recall indicates that many actual survivors were missed. Tuning produced a higher selection-CV score, but did not produce higher accuracy than the simple comparator on this test split.

**Output status:** These metrics come from the author's refreshed run after the title-extraction correction. The uploaded notebook contains all 18 code cells executed in sequence, with no saved error outputs. The saved outputs were inspected for consistency; this review did not independently rerun training.

## Run locally

The saved run recorded Python **3.13.15**. The core package versions in `requirements.txt` match that run. Use a separate environment for this project.

Clone the repository and enter the project folder:

```bash
git clone https://github.com/saad-khanzada/machine-learning-projects.git
cd machine-learning-projects/titanic-survival-classification
python -m venv .venv
```

Activate the environment:

| Shell | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Install dependencies and open the notebook:

```bash
python -m pip install -r requirements.txt
python -m notebook titanic_classification_project.ipynb
```

Select **Run All**. Keep `titanic_dataset.csv` in the project folder. If the working directory differs, set `DATA_PATH` to the CSV's location. Training uses the CPU; no GPU or API key is required.

## Run in Google Colab

1. Open the Colab badge above.
2. Download [titanic_dataset.csv](https://raw.githubusercontent.com/saad-khanzada/machine-learning-projects/main/titanic-survival-classification/titanic_dataset.csv) and upload it through Colab's Files panel.
3. Select **Runtime → Run all**. If no CSV is found, the loading cell prompts for an upload.

Opening the notebook in Colab does not automatically copy the repository's CSV into the runtime. Colab package versions may differ from the recorded environment; the optional setup cell lists the pinned core versions.

## Files and optional exports

| File | Purpose |
|---|---|
| `titanic_classification_project.ipynb` | Analysis, training, saved outputs, and prediction example |
| `titanic_dataset.csv` | Passenger data used in the experiment |
| `requirements.txt` | Core analysis packages and local notebook interface |
| `README.md` | Project guide |

To predict from another compatible CSV, set `NEW_PASSENGERS_PATH` in the prediction cell. To save CV results, test metrics, passenger predictions, and a JSON run record, set `EXPORT_RESULTS = True`. Exports are disabled by default.

## Limitations

This is a small historical dataset with a single test split and prior dataset exploration. Related family or ticket groups can cross the random split. CV scores are used for selection and can be optimistic after comparing alternatives. Predicted probabilities are uncalibrated, and no external validation is included.

The project demonstrates a classification workflow; the reported results do not establish performance on another population.
