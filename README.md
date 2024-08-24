# ReneWind: Predictive Maintenance in Wind Energy

## Business Overview
ReneWind is a company working on improving the machinery / processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. 

## Objective
The task at hand is to build various classification models, tune them, and find the best one that will help identify failures so that the generators could be repaired before failing/breaking to reduce the overall maintenance cost. 

## Key Skills
`Feature Engineering`
`Cross Validation`
`ML Pipeline`
`Hyperparameter Tuning`

## Python Libraries

- **Pandas** (`import pandas as pd`)
- **NumPy** (`import numpy as np`)
- **Matplotlib** (`import matplotlib.pyplot as plt`)
- **Seaborn** (`import seaborn as sns`)
- **Scikit-learn (sklearn)**
  - Missing value imputation: `SimpleImputer` (`from sklearn.impute import SimpleImputer`)
  - Model building:
    - `LogisticRegression` (`from sklearn.linear_model import LogisticRegression`)
    - `DecisionTreeClassifier` (`from sklearn.tree import DecisionTreeClassifier`)
    - `AdaBoostClassifier`, `GradientBoostingClassifier`, `RandomForestClassifier`, `BaggingClassifier` (`from sklearn.ensemble import ...`)
    - `XGBClassifier` (`from xgboost import XGBClassifier`)
  - Metrics and data splitting:
    - `metrics` (`from sklearn import metrics`)
    - `train_test_split`, `StratifiedKFold`, `cross_val_score` (`from sklearn.model_selection import ...`)
    - Metrics: `f1_score`, `accuracy_score`, `recall_score`, `precision_score`, `confusion_matrix`, `roc_auc_score`, `ConfusionMatrixDisplay` (`from sklearn.metrics import ...`)
  - Preprocessing:
    - `StandardScaler`, `MinMaxScaler`, `OneHotEncoder` (`from sklearn.preprocessing import ...`)
  - Model tuning:
    - `GridSearchCV`, `RandomizedSearchCV` (`from sklearn.model_selection import ...`)
  - Pipeline creation:
    - `Pipeline` (`from sklearn.pipeline import Pipeline`)
    - `ColumnTransformer` (`from sklearn.compose import ColumnTransformer`)
- **Imbalanced-learn (imblearn)**
  - Oversampling: `SMOTE` (`from imblearn.over_sampling import SMOTE`)
  - Undersampling: `RandomUnderSampler` (`from imblearn.under_sampling import RandomUnderSampler`)
- **XGBoost** (`from xgboost import XGBClassifier`)
- **Pandas Configuration**:
  - Display all columns (`pd.set_option("display.max_columns", None)`)
  - Suppress scientific notation (`pd.set_option("display.float_format", lambda x: "%.3f" % x)`)
- **Warnings** (`import warnings`)
  - Suppress warnings (`warnings.filterwarnings("ignore")`)

## Project Flowchart

<img width="916" alt="image" src="https://github.com/user-attachments/assets/8cfae7bc-a14a-4df0-8cd4-223d08b1715e">

## Charts

![image](https://github.com/user-attachments/assets/4a3c4c35-e7bb-4795-860e-f65834122a81)

![image](https://github.com/user-attachments/assets/aff25e18-d76f-4970-9a8b-be8bf2ec1c72)

![image](https://github.com/user-attachments/assets/395312b6-5d26-468e-b3c9-d2c4dac7ed07)

![image](https://github.com/user-attachments/assets/412922ae-d895-4a56-b43b-a0021f58a969)

![image](https://github.com/user-attachments/assets/252a8293-89e0-4ed3-bdb7-ec3c1cd75e68)

![image](https://github.com/user-attachments/assets/05a96e49-0545-4e0e-8bab-ec8207d24740)

