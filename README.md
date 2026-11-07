# Shopper Intent Lab: Predicting Online Purchases

**Live app:** REPLACE_WITH_STREAMLIT_URL

## a. Problem statement

Most visits to an online store end without a purchase. The goal of this project is to predict, from the behaviour recorded during a browsing session, whether that session will end in a purchase (`Revenue` = TRUE or FALSE). This is a binary classification problem.

Five classifiers are trained on the same data and compared on six evaluation metrics. A Streamlit web application then loads the saved models and scores an uploaded test CSV without any retraining.

## b. Dataset description

| Property | Value |
| --- | --- |
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Reference | Sakar, C. and Kastro, Y. (2018). DOI: 10.24432/C5F88Q |
| Instances | 12,330 (12,205 after removing 125 exact duplicates) |
| Input features | 17 (10 numeric, 7 categorical) |
| Target | `Revenue`: TRUE (purchase) or FALSE (no purchase) |
| Class balance | About 15.5% purchase and 84.5% no purchase |
| Missing values | None |

**Features**

- `Administrative`, `Informational`, `ProductRelated`: number of pages of each type visited in the session.
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`: time spent on each page type.
- `BounceRates`, `ExitRates`, `PageValues`: web-analytics measures for the pages visited.
- `SpecialDay`: closeness of the visit to a special day such as Valentine's Day.
- `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`: categorical session details. The dataset has no sessions for January or April.

**Preparation**

1. Removed exact duplicate rows before splitting, so that a row cannot appear in both the training and test sets.
2. Made a stratified 80/20 split: 9,764 training rows and 2,441 test rows. The test rows are saved as `test_data.csv`.
3. Applied `log1p` and standard scaling to the numeric features, because they have long right tails.
4. One-hot encoded the categorical features. `OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as integers, but they are ID codes with no order, so they are also treated as categories.
5. Tuned each model with 3-fold stratified cross-validation on the training rows only. The best mean F1 was chosen, with mean AUC used to separate ties. Preprocessing is saved inside each model's pipeline.

## c. GitHub repository link

REPLACE_WITH_GITHUB_URL

```
shopper-intent-classifier/
|-- app.py                 Streamlit application
|-- data_explorer.py       Data explorer tab of the app
|-- train_models.py        Training and evaluation script
|-- shopper_core.py        Shared schema, validation and metric code
|-- assets/style.css       App styling
|-- requirements.txt
|-- README.md
|-- test_data.csv          Held-out test data with the target column
|-- data/                  Original UCI dataset
|-- model/                 Five saved pipelines (.joblib) and training_summary.json
|-- .streamlit/            App theme
```

## d. Models used

All metrics below are measured on the 2,441 held-out test rows, with "purchase" as the positive class. AUC uses predicted probabilities, except for the SVM, which uses its decision function.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8869 | 0.9276 | 0.6082 | 0.7801 | 0.6835 | 0.6230 |
| Decision Tree | 0.8820 | 0.9060 | 0.5907 | 0.8010 | 0.6800 | 0.6203 |
| KNN | 0.9045 | 0.9186 | 0.7790 | 0.5445 | 0.6410 | 0.6005 |
| Naive Bayes | 0.8603 | 0.8748 | 0.5439 | 0.6649 | 0.5984 | 0.5186 |
| Support Vector Machine | 0.8873 | 0.9338 | 0.6085 | 0.7853 | 0.6857 | 0.6259 |

Predicting "no purchase" for every session would already reach 84.35% accuracy on this test set, so accuracy alone says little here. F1, MCC and AUC are more informative.

**Selected hyperparameters**

| ML Model Name | Selected settings | CV F1 | CV AUC |
| --- | --- | --- | --- |
| Logistic Regression | C=0.003, class_weight=balanced | 0.6665 | 0.9097 |
| Decision Tree | max_depth=2, min_samples_leaf=1, class_weight=balanced | 0.6594 | 0.8953 |
| KNN | n_neighbors=21, weights=uniform | 0.6024 | 0.9019 |
| Naive Bayes | GaussianNB, var_smoothing=1.0 | 0.5886 | 0.8666 |
| Support Vector Machine | RBF kernel, C=3, gamma=0.003, class_weight=balanced | 0.6652 | 0.9155 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Second-best F1 (0.6835) and MCC (0.6230), and it catches 78% of buyers. Training F1 (0.6674) almost equals CV F1 (0.6665), so it shows no sign of overfitting. `PageValues` has by far the largest coefficient (1.157). It trains in seconds, which makes it the best value for its cost. |
| Decision Tree | Depth 2 was selected, and it has the highest recall (0.8010: 306 of 382 buyers). Both second-level splits give the same class as their parent, so its predictions are effectively one threshold on `PageValues`. That single rule nearly matches Logistic Regression on F1 (0.6800), which shows how much this one feature carries. With only four leaves, its scores are coarse, which explains the lower AUC (0.9060). |
| KNN | Highest accuracy (0.9045) and precision (0.7790), with only 59 false alarms, but the lowest recall (0.5445): it misses 174 of 382 buyers. It has no class weighting, so the majority class dominates each neighbourhood. It also has the largest gap between training F1 (0.6419) and CV F1 (0.6024), and its saved file is the largest because it stores the training data. |
| Naive Bayes | Weakest overall, with the lowest F1 (0.5984), MCC (0.5186) and AUC (0.8748). Its independence assumption does not hold here: `BounceRates` and `ExitRates` correlate at 0.90, and `ProductRelated` and its duration at 0.86. One-hot columns are also poorly described by a Gaussian. The search chose very heavy smoothing (var_smoothing=1.0). It is the fastest model to train. |
| Support Vector Machine | Best on the test set for AUC (0.9338), F1 (0.6857) and MCC (0.6259). Its lead over Logistic Regression is small: 2 more buyers caught and 1 more false alarm across 2,441 rows. It is also the slowest model by far, taking most of the total training time. |
| Overall, Winner for your dataset? | **Support Vector Machine, by a narrow margin.** It leads on the balanced metrics and on AUC in both the test set and cross-validation. The gap on F1 is within noise, as Logistic Regression is slightly ahead on CV F1 (0.6665 vs 0.6652). If training cost or interpretability mattered, Logistic Regression would be the practical choice. |

**Limitations**

- All metrics use each model's default decision threshold. Adjusting the threshold would trade precision against recall, and could change the ranking on those metrics.
- The SVM search left out larger values such as C=10 with gamma=0.03, because an earlier grid including them took 54 minutes. The selected C=3 is at the edge of the final grid, although CV F1 changed very little between neighbouring settings.

## Running the project

Python 3.12 is recommended, because the versions in `requirements.txt` are pinned to match the saved models.

```
pip install -r requirements.txt
python train_models.py      (optional: rebuilds model/ and test_data.csv)
streamlit run app.py
```

## Using the app

1. Upload a test CSV in the sidebar, or use the bundled `test_data.csv`. The file must contain all 17 features and the `Revenue` column.
2. Choose a model from the dropdown.
3. Read the six metrics, the confusion matrix and the classification report under **Selected model**.
4. Compare every model on the same file, including ROC curves, under **All five compared**.
5. Under **Data explorer**, see a description of the uploaded CSV: a column dictionary, class balance, feature distributions, correlations and a raw-data preview.
6. Inspect or download individual predictions under **Row-level predictions**.
