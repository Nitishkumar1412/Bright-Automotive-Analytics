# 🚗 Bright Automotive — EDA & Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

An end-to-end **Data Science and Machine Learning project** for Bright Automotive Company focused on understanding customer profiles, analyzing purchase behavior, predicting vehicle prices, and classifying the type of vehicle a customer is likely to purchase.

---

## 📌 Project Overview

Bright Automotive Company has collected customer information from previous inquiries and purchases. The dataset contains demographic, professional, financial, and vehicle-related information.

The goal of this project is to use this data to:

* 🔍 Explore customer and purchase patterns
* 🧐 Identify data quality issues
* 📊 Perform Exploratory Data Analysis (EDA)
* 🧹 Clean and preprocess the dataset
* 💰 Predict the **price of a vehicle**
* 🚘 Classify the **vehicle type / Make**
* ⚖️ Compare multiple Machine Learning algorithms
* ✅ Apply Cross-Validation
* 🎛️ Perform Hyperparameter Tuning

The project follows a complete workflow from **raw data → EDA → preprocessing → machine learning → model evaluation**.

---

## 🎯 Objectives

### 1. 📊 Exploratory Data Analysis

The project investigates:

* Customer demographics
* Gender
* Profession
* Marital status
* Education
* Number of dependents
* Personal and house loans
* Partner employment
* Salary and partner salary
* Total household salary
* Vehicle price
* Vehicle make/type

The notebook also analyzes duplicate records, missing values, outliers, and relationships between different variables.

### 2. 📈 Regression

The regression task predicts the **Price** of a vehicle using customer demographic and financial information.

The following regression algorithms were evaluated:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

Evaluation metrics include:

* RMSE
* R² Score

### 3. 🏷️ Classification

The classification task predicts the **Make / vehicle type** based on customer and financial characteristics.

Classification algorithms used:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

Evaluation includes:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

## 📊 Dataset

The dataset contains **1,586 records and 14 features**.

### 🧾 Features

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| `Age`              | Customer age                              |
| `Gender`           | Customer gender                           |
| `Profession`       | Customer profession                       |
| `Marital_status`   | Marital status                            |
| `Education`        | Education level                           |
| `No_of_Dependents` | Number of dependents                      |
| `Personal_loan`    | Whether the customer has a personal loan  |
| `House_loan`       | Whether the customer has a house loan     |
| `Partner_working`  | Whether the customer's partner is working |
| `Salary`           | Customer salary                           |
| `Partner_salary`   | Partner's salary                          |
| `Total_salary`     | Combined salary                           |
| `Price`            | Vehicle price                             |
| `Make`             | Vehicle type / make                       |

The dataset contains missing values and inconsistent entries that are addressed during preprocessing. For example, the notebook identifies missing values in `Gender`, `Profession`, `Salary`, and `Partner_salary`, as well as inconsistent categorical values such as `Femal` and `?`.

---

## 🔎 Exploratory Data Analysis

The EDA process includes:

### 🧪 Basic Analysis

* Dataset shape
* Data types
* First few records
* Statistical summary
* Duplicate records
* Missing-value analysis

### ❓ Missing Values

The original dataset contains:

* `Gender`: 53 missing values
* `Profession`: 11 missing values
* `Salary`: 13 missing values
* `Partner_salary`: 106 missing values

The notebook also calculates the percentage of missing values for each feature.

### ⚠️ Data Quality Issues

Some issues identified include:

* Incorrect gender entries
* `?` values
* Missing categorical values
* Missing numerical values
* Duplicate records
* Potential age outliers
* Potential price outliers

The notebook identifies **5 duplicate rows** and also investigates unusual ages such as 120.

---

## 🧹 Data Preprocessing

The preprocessing workflow includes:

* Handling missing values
* Correcting categorical values
* Treating outliers
* Encoding categorical variables
* Train-validation split
* Feature scaling

Categorical variables are processed using techniques including:

* Ordinal Encoding
* Nominal Encoding
* One-Hot Encoding

Numerical variables are standardized using `StandardScaler`.

---

## 🤖 Regression Results

The initial model comparison produced the following validation results:

| Model             | Validation RMSE | Validation R² |
| ----------------- | --------------: | ------------: |
| Linear Regression |         7983.50 |        0.6458 |
| KNN               |         6694.68 |        0.7509 |
| Decision Tree     |         8220.73 |        0.6244 |
| Random Forest     |     **5744.63** |    **0.8166** |
| AdaBoost          |         6536.69 |        0.7625 |
| Gradient Boosting |         5862.05 |        0.8090 |
| XGBoost           |         6307.91 |        0.7789 |

After 5-fold cross-validation, Random Forest achieved a validation RMSE of approximately **5762.84** and R² of approximately **0.8154**.

The notebook then applied GridSearchCV to tune the Random Forest model.

### 🌲 Tuned Random Forest

Best hyperparameters:

```text
max_depth = 10
max_leaf_nodes = 19
min_samples_split = 11
n_estimators = 130
```

Performance:

```text
Train RMSE = 5218.64
Train R²   = 0.8536
Validation RMSE = 5910.66
Validation R²   = 0.8059
```

The notebook also performs XGBoost hyperparameter tuning.

---

## 🚗 Classification Results

The classification models were used to predict the vehicle `Make`.

Initial validation accuracy:

| Model               | Validation Accuracy |
| ------------------- | ------------------: |
| Logistic Regression |              72.24% |
| KNN                 |              73.82% |
| Decision Tree       |              83.91% |
| Random Forest       |              80.44% |
| AdaBoost            |              66.25% |
| Gradient Boosting   |          **85.49%** |
| XGBoost             |              83.91% |

5-fold cross-validation was also performed to evaluate model consistency.

The notebook reports a mean CV accuracy of approximately **80.46%** for Gradient Boosting and approximately **80.69%** for XGBoost.

### 🚀 Tuned Gradient Boosting

GridSearchCV was used to tune the Gradient Boosting classifier.

Best hyperparameters:

```text
learning_rate = 0.1
max_depth = 3
n_estimators = 100
```

Performance:

```text
Train Accuracy = 92.17%
Validation Accuracy = 85.80%
```

The classification report for the validation dataset also shows an overall accuracy of approximately **85%** before tuning.

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📉 Matplotlib
* 🎨 Seaborn
* 🤖 Scikit-learn
* ⚡ XGBoost
* 📓 Google Colab / Jupyter Notebook

---

## 📁 Repository Structure

```text
bright-automotive-eda-ml/
│
├── bright_automotive_company.csv
├── Project.ipynb
├── README.md
│
└── generated/
    ├── cleaned_data.csv
    ├── encoded_data.csv
    └── scaled_data.csv
```

> 💡 The `generated` folder is optional and can be added if the processed datasets are included in the repository.

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Missing Value Treatment
     ↓
Outlier Treatment
     ↓
Exploratory Data Analysis
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train / Validation Split
     ↓
Regression Modeling
     ↓
Classification Modeling
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
```

---

## 💼 Business Use Case

This project can help an automotive company understand customer profiles and support data-driven decision-making.

Potential applications include:

* 👥 Customer segmentation
* 🚙 Vehicle recommendation
* 💵 Price estimation
* 📣 Sales strategy planning
* 🛒 Understanding customer purchasing patterns
* 💸 Identifying relationships between income and vehicle choice
* 🎯 Supporting personalized vehicle pricing

---

## 📌 Key Takeaways

* The dataset contains **1,586 customer records and 14 features**.
* Several data-quality issues were identified and treated.
* EDA was performed on demographic, professional, financial, and vehicle variables.
* Multiple regression algorithms were compared for vehicle price prediction.
* Multiple classification algorithms were compared for vehicle-type prediction.
* Cross-validation was used to assess model performance.
* GridSearchCV was used for hyperparameter tuning.
* The project demonstrates a complete practical Machine Learning workflow.

---

## 👨‍💻 Project

**Bright Automotive — Customer Purchase Behavior Analysis**

This repository contains the dataset, exploratory analysis, preprocessing workflow, machine learning experiments, model comparison, and tuning performed as part of the project.
