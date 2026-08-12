# Heart Disease Dataset

## Overview
This dataset contains clinical and physiological features collected from cardiac patients to predict the presence of heart disease. In the context of the **Cardiac Patient Monitoring System** project, this dataset is utilized for exploratory data analysis (EDA), statistical evaluation, domain feature engineering, and supervised classification modeling to predict cardiac disease risk.

- **File Name:** `heart.csv`
- **File Path:** [heart.csv](file:///c:/Users/Amr-Sheqwara/Desktop/AI_ML_Internship/Projects/cardiac_monitoring_project/Data/heart.csv)
- **Total Records (Rows):** 918
- **Total Features (Columns):** 12 (11 clinical features + 1 target variable)
- **File Format:** Comma-Separated Values (CSV)

---

## Target Variable

- **Variable Name:** `HeartDisease`
- **Definition:** Indicates the presence or absence of significant heart disease.
- **Data Type:** Integer / Binary (`0` or `1`)
- **Distribution:**
  - `0` (Normal / No Heart Disease): 410 records (~44.66%)
  - `1` (Heart Disease): 508 records (~55.34%)
- **Class Balance Note:** The target variable is relatively well-balanced compared to typical medical datasets, allowing for balanced cross-validation and standard classification metric comparisons.

---

## Data Dictionary

| Column Name | Data Type | Description | Permitted Values / Range | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `Age` | Integer / Continuous | Age of the patient in years | 28 to 77 | 0 (0.0%) |
| `Sex` | Categorical (String) | Biological sex of the patient | `M` (Male), `F` (Female) | 0 (0.0%) |
| `ChestPainType` | Categorical (String) | Chest pain classification type | `TA` (Typical Angina), `ATA` (Atypical Angina), `NAP` (Non-Anginal Pain), `ASY` (Asymptomatic) | 0 (0.0%) |
| `RestingBP` | Integer / Continuous | Resting blood pressure (mm Hg) | 0 to 200 | 0 (0.0%) |
| `Cholesterol` | Integer / Continuous | Serum cholesterol level (mm/dl) | 0 to 603 | 0 (0.0%) |
| `FastingBS` | Integer / Binary | Fasting blood sugar indicator | `1` if FastingBS > 120 mg/dl, `0` otherwise | 0 (0.0%) |
| `RestingECG` | Categorical (String) | Resting electrocardiogram results | `Normal` (Normal), `ST` (ST-T wave abnormality), `LVH` (Left Ventricular Hypertrophy) | 0 (0.0%) |
| `MaxHR` | Integer / Continuous | Maximum heart rate achieved during exercise (bpm) | 60 to 202 | 0 (0.0%) |
| `ExerciseAngina` | Categorical (String) | Exercise-induced angina | `Y` (Yes), `N` (No) | 0 (0.0%) |
| `Oldpeak` | Float / Continuous | ST depression induced by exercise relative to rest | -2.6 to 6.2 | 0 (0.0%) |
| `ST_Slope` | Categorical (String) | Slope of the peak exercise ST segment | `Up` (Upsloping), `Flat` (Flat), `Down` (Downsloping) | 0 (0.0%) |
| `HeartDisease` | Integer / Binary | Target classification variable | `0` = Normal, `1` = Heart Disease | 0 (0.0%) |

---

## Detailed Clinical Feature Descriptions

1. **`Age`**: Patient age in years.
2. **`Sex`**: Binary sex classification (`M` for Male: 725 records, `F` for Female: 193 records).
3. **`ChestPainType`**: Four distinct clinical presentations:
   - `ASY`: Asymptomatic (highest frequency, strongly associated with cardiac risk).
   - `NAP`: Non-Anginal Pain.
   - `ATA`: Atypical Angina.
   - `TA`: Typical Angina.
4. **`RestingBP`**: Blood pressure at rest in mm Hg. Note: Contains 1 physiologically invalid value of `0` requiring handling during data cleaning.
5. **`Cholesterol`**: Serum cholesterol in mm/dl. Note: Contains 172 records with value `0`, which represent unrecorded/missing measurements requiring median or model-based imputation.
6. **`FastingBS`**: Fasting blood sugar threshold classification (`1` if blood sugar exceeds 120 mg/dl, `0` otherwise).
7. **`RestingECG`**: Resting ECG findings:
   - `Normal`: Normal baseline ECG.
   - `ST`: Presence of ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV).
   - `LVH`: Showing probable or definite left ventricular hypertrophy by Estes' criteria.
8. **`MaxHR`**: Maximum heart rate recorded during cardiac stress testing.
9. **`ExerciseAngina`**: Binary indicator of whether physical exercise induced angina pectoris (`Y` / `N`).
10. **`Oldpeak`**: Numeric measurement of ST segment depression recorded on electrocardiogram after exercise compared to rest.
11. **`ST_Slope`**: The geometric trajectory of the ST segment during peak exercise (`Up`, `Flat`, `Down`).
12. **`HeartDisease`**: Binary target label (`1` indicates presence of heart disease, `0` indicates normal condition).

---

## Data Preparation & Preprocessing Guidelines

- **Missing / Invalid Value Treatment:**
  - `Cholesterol == 0`: 172 records with zero cholesterol must be replaced with `NaN` and imputed (e.g., using median strategy or KNN imputer) in preprocessing.
  - `RestingBP == 0`: 1 record with zero blood pressure should be imputed with the median resting blood pressure.
- **Categorical Feature Encoding:**
  - Binary nominals: `Sex` (`M`/`F`) and `ExerciseAngina` (`Y`/`N`) mapped to numeric binary (`0`/`1`).
  - Multi-category nominals: `ChestPainType`, `RestingECG`, and `ST_Slope` processed using `OneHotEncoder(handle_unknown='ignore')`.
- **Feature Scaling:**
  - Continuous numerical features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) require standardization via `StandardScaler()` when applying Logistic Regression, Support Vector Classifiers, and distance-based estimators.
- **Pipeline Integration:**
  - Package imputation, encoding, and scaling into a unified Scikit-learn `ColumnTransformer` to avoid data leakage during cross-validation.
