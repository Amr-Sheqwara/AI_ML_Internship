# Cardiac Patient Monitoring System 

## Overview

The **Cardiac Patient Monitoring System** is an individual AI & Machine Learning project developed as part of the BinX Tech AI & ML Internship program. The project focuses on building a curriculum-aligned machine-learning analysis for public cardiac and stroke health data.
  
The end-to-end workflow covers data loading, thorough data cleaning and preparation, exploratory data analysis (EDA) with descriptive statistics and visualizations, supervised classification modeling with cross-validation and standard metrics, and automated preprocessing pipelines.

---

## Project Objectives

1.  **Environment and Data Inspection:** Set up a reproducible Python/Jupyter environment, inspect dataset columns and data types, and document the target classification variable.

2.  **Data Preparation:** Use Pandas and NumPy to handle missing values, duplicates, invalid values, categorical variables, and basic feature cleanup.

3.  **Exploratory Data Analysis (EDA):** Perform descriptive statistics and Matplotlib visualizations to study feature distributions, relationships, class balance, correlations, and potential outliers.

4.  **Supervised Classification:** Define a supervised learning problem to predict stroke risk, train a baseline classifier (e.g., Logistic Regression), and compare it against additional Scikit-learn classifiers.

5.  **Model Evaluation:** Evaluate models using train/test splitting, stratified cross-validation, confusion matrices, and relevant classification metrics (Accuracy, Precision, Recall, F1-score, and ROC-AUC), explaining false positives and false negatives in plain language.

6.  **Feature Engineering & Pipelines:** Construct reusable Scikit-learn Pipelines combining preprocessing, encoding, scaling, and classification models for consistent execution.

---

## Scope & Boundaries

### In Scope

- Python environment utilizing Jupyter Notebook, NumPy, Pandas, Matplotlib, and Scikit-learn.

- Data cleaning, descriptive statistics, exploratory data analysis, correlation analysis, and charts.

- Supervised binary classification using Scikit-learn (baseline model and comparison models).

- Model evaluation via train/test splitting, cross-validation, confusion matrices, and metrics.

- Preprocessing and feature engineering encapsulated in reusable Scikit-learn Pipelines.

### Out of Scope

- Deep learning architectures, neural networks, Large Language Models (LLMs), cloud ML platforms, MLOps orchestration, FastAPI deployment, or production serving infrastructure.

- Clinical diagnosis, medical treatment recommendations, emergency instructions, or use of identifiable patient data.

- Advanced explainability packages or techniques not covered in the training curriculum.

- Dependencies on external production systems or external student projects.

---

## Project Structure

```text
cardiac_monitoring_project/

├── Data/

│ ├── README.md # Dataset documentation and data dictionary

│ └── healthcare-dataset-stroke-data.csv # Public healthcare stroke dataset (5,110 records)

├── Notebooks/

│ └── Cardiac_Patient_Monitoring_System.ipynb # End-to-end analysis and modeling notebook

├── Outputs/ # Generated plots, evaluation matrices, and artifacts

├── requirements.txt # Python environment dependencies

└── README.md # Project documentation and execution guide
  ```

---

## Setup and Execution Guide

### Prerequisites

- Python 3.10 or higher

- PowerShell / Command Line

### 1. Activate Virtual Environment

From the workspace root directory:

```powershell

.\.venv\Scripts\Activate.ps1

```

### 2. Install Dependencies

Install all required packages from the requirements file:

```powershell

pip install -r requirements.txt

```

### 3. Launch the Jupyter Environment

```powershell

jupyter notebook

```
