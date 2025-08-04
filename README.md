# 🤖 Machine Learning Project: Predictive Modeling Pipeline

This project follows the **machine learning life cycle** from problem definition through model optimization. The goal was to explore and implement a complete supervised learning pipeline using a dataset of choice, leveraging classification techniques to solve a predictive problem.

---

## 📌 Project Workflow

The following steps were completed in accordance with the machine learning process:

### 1. 🧱 Build Your DataFrame
- Selected an appropriate dataset and imported it into a structured `pandas` DataFrame for further analysis and modeling.

### 2. 🎯 Define Your ML Problem
- Clearly defined the prediction task as a binary classification problem.
- Identified the label (target variable) and selected meaningful features based on domain knowledge and exploratory analysis.

### 3. 📊 Exploratory Data Analysis (EDA)
- Inspected the data distribution, feature types, and potential signal strength.
- Analyzed outliers and missing values.
- Investigated class imbalance and correlations between features and the target.

### 4. 📐 Define Your Project Plan
- Chose **logistic regression** as the base model due to interpretability and performance for binary classification.
- Planned feature engineering, evaluation metrics, and performance tuning approaches.

---

## 🛠️ Implementation Steps

### 🔧 Data Preprocessing
- **Outlier Handling**: Applied **winsorization** to cap extreme values.
- **Missing Values**: Filled missing numerical entries using **column means**.
- **Categorical Features**: Used **one-hot encoding** on object-type columns.
- **Class Imbalance**: Addressed via `class_weight='balanced'` in logistic regression.

### 🧪 Model Training & Evaluation
- Trained a **baseline logistic regression model** and evaluated its performance.
- Calculated **log loss** and **accuracy scores** across a range of `C` values (regularization strengths).
- Plotted **log loss vs C** and **accuracy vs C** to identify promising hyperparameter regions.

### 🔍 Hyperparameter Optimization
- Performed **GridSearchCV** to find the optimal `C` value.
- Evaluated multiple models trained with different `C` values selected via:
  - Lowest log loss
  - Highest accuracy
  - Grid search best estimate

### 📈 Visualization & Metrics
- Plotted **Precision-Recall curves** and **ROC curves** for each tuned model.
- Compared AUC scores across different configurations.

### ⭐ Feature Selection
- Applied `SelectKBest` to isolate the **top 10 predictive features** based on univariate statistical tests.
- Re-evaluated AUC after feature reduction to assess model performance on a reduced feature set.

---

## ✅ Results Summary

| Evaluation Method       | Best C Value |  AUC Score  |
|-------------------------|--------------|-------------|
| Log Loss Minimization   |    *0.1*     |  *0.90531*  |
| Accuracy Maximization   |    *10*      |  *0.90500*  |
| GridSearchCV Best Value |    *1.0*     |  *0.90486*  |
| SelectKBest Top 10      |    *N/A*     |  *0.89274*  |

---

## 💾 Tech Stack

- Python 3.x
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for modeling, evaluation, and selection
- `matplotlib`, `seaborn` for plotting and visualization
- `scipy.stats` for winsorization
- `joblib` (optional) for model persistence
