# 🏠 ML Assignment 4 – Regression & Evaluation Metrics
### California Housing Price Prediction using Supervised Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project applies and evaluates **five regression algorithms** on the California Housing dataset to predict **median house prices**. It covers the complete machine learning pipeline — from data exploration and preprocessing to model evaluation, cross-validation, hyperparameter tuning, and final model selection.

The assignment is structured to demonstrate:
- Understanding of different regression techniques and their trade-offs
- Application of standard evaluation metrics (MSE, MAE, R²)
- Model improvement through systematic hyperparameter tuning
- Justification of the best model backed by quantitative evidence

---

## 📁 Repository Structure

```
ML-Assignment-4-Regression/
│
├── ML_Assignment_4_Regression.ipynb   # Main Jupyter Notebook — all code, markdown, and plots
├── README.md                          # Project documentation (this file)
│
└── (outputs are rendered inline in the notebook)
```

---

## 🎯 Objective

> Evaluate the understanding of regression techniques in supervised learning by applying them to a real-world dataset and analyzing their performance through comprehensive evaluation metrics.

---

## 📊 Dataset — California Housing

| Property | Details |
|---|---|
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Total Samples** | 20,640 |
| **Features** | 8 numerical features |
| **Target Variable** | `MedHouseVal` — Median House Value (in $100,000s) |
| **Missing Values** | None |
| **Origin** | 1990 California Census data |

### Feature Descriptions

| Feature | Full Name | Description |
|---|---|---|
| `MedInc` | Median Income | Median income of households in the block (in $10,000s) |
| `HouseAge` | House Age | Median age of houses in the block (years) |
| `AveRooms` | Average Rooms | Average number of rooms per household |
| `AveBedrms` | Average Bedrooms | Average number of bedrooms per household |
| `Population` | Population | Total population of the block |
| `AveOccup` | Average Occupancy | Average number of household members |
| `Latitude` | Latitude | Geographic latitude of the block |
| `Longitude` | Longitude | Geographic longitude of the block |
| `MedHouseVal` | **TARGET** | Median house value for households in the block |

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ML-Assignment-4-Regression.git
cd ML-Assignment-4-Regression
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate          # On macOS/Linux
venv\Scripts\activate             # On Windows
```

### 3. Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy jupyter
```

### 4. Launch the Notebook

```bash
jupyter notebook ML_Assignment_4_Regression.ipynb
```

### 5. Run All Cells

In Jupyter: **Kernel → Restart & Run All**

> ⚠️ **Note:** Hyperparameter tuning cells (GridSearchCV / RandomizedSearchCV) may take **5–15 minutes** depending on your hardware. This is expected — the notebook will complete successfully.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.8+ | Core language |
| `scikit-learn` | ≥ 1.0 | ML models, metrics, preprocessing |
| `pandas` | ≥ 1.3 | Data manipulation |
| `numpy` | ≥ 1.21 | Numerical operations |
| `matplotlib` | ≥ 3.4 | Plotting |
| `seaborn` | ≥ 0.11 | Statistical visualizations |
| `scipy` | ≥ 1.7 | Random distributions for RandomizedSearchCV |
| `jupyter` | ≥ 1.0 | Notebook environment |

---

## 🧩 Assignment Structure — Key Components

---

### Section 1 — Data Loading & Preprocessing

**What's covered:**
- Loading the dataset using `fetch_california_housing(as_frame=True)`
- Converting to a pandas DataFrame
- Checking for missing values (confirmed: none present)
- **Exploratory Data Analysis (EDA):**
  - Descriptive statistics (mean, std, min, max, quartiles)
  - Feature distribution histograms for all 9 columns
  - Correlation heatmap to identify relationships between features
  - Scatter plots of top correlated features vs. the target

**Feature Scaling:**
- Method used: **StandardScaler (Z-score Standardization)**
- Formula: `z = (x - μ) / σ`
- Justification:
  - Features have vastly different scales (e.g., `Population` ranges 0–35,000; `MedInc` ranges 0–15)
  - SVR and Linear Regression are highly sensitive to feature magnitude — standardization prevents domination by high-scale features
  - More robust to outliers compared to Min-Max normalization
  - Scaler is **fit only on training data** to prevent data leakage

**Train/Test Split:** 80% training / 20% testing (`random_state=42`)

---

### Section 2 — Regression Algorithm Implementation

Five algorithms are implemented and compared:

#### 1. 📐 Linear Regression
- Models target as a linear combination of input features: `ŷ = β₀ + β₁x₁ + … + βₙxₙ`
- Minimizes Residual Sum of Squares (OLS method)
- **Role:** Strong interpretable baseline

#### 2. 🌳 Decision Tree Regressor
- Recursively partitions the feature space via binary splits (minimizing MSE)
- Predicts mean target value within each leaf
- **Strength:** Captures non-linear patterns and feature interactions
- **Weakness:** Prone to overfitting without depth constraints

#### 3. 🌲 Random Forest Regressor
- Ensemble of Decision Trees trained on bootstrap samples (bagging)
- Each tree uses a random feature subset at each split
- Final prediction = average across all trees
- **Strength:** Reduces variance significantly vs. a single tree; robust to outliers

#### 4. 🚀 Gradient Boosting Regressor
- Builds trees sequentially — each new tree corrects residual errors of the prior ensemble
- Uses gradient descent in function space
- **Strength:** Best accuracy on tabular data; handles complex non-linearities

#### 5. 🔵 Support Vector Regressor (SVR)
- Finds a hyperplane that fits within an ε-insensitive tube around the data
- Uses RBF kernel to handle non-linear relationships
- **Strength:** Effective in high-dimensional spaces
- **Requirement:** Needs feature scaling (already applied)

---

### Section 3 — Model Evaluation & Comparison

**Metrics used:**

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | `mean((y - ŷ)²)` | Average squared error; penalizes large errors heavily |
| **RMSE** | `√MSE` | Same unit as target; interpretable error magnitude |
| **MAE** |  `mean((y - ŷ))`  | Average absolute error; robust to outliers |
| **R²** | `1 - SS_res/SS_tot` | Proportion of variance explained; 1.0 = perfect |

**Visualizations included:**
- Horizontal bar charts for MSE, MAE, and R² across all models
- Actual vs. Predicted scatter plots for all 5 models (with R² annotated)

---

### Section 4 — Cross-Validation & Hyperparameter Tuning

#### Cross-Validation
- **Method:** 5-Fold KFold (`shuffle=True`, `random_state=42`)
- All models evaluated on scaled training data
- Reports: Mean R², Std R², Min R², Max R² per model

#### Hyperparameter Tuning

| Model | Method | Key Parameters Tuned |
|---|---|---|
| **Ridge Regression** | GridSearchCV | `alpha` (regularization strength) |
| **Decision Tree** | GridSearchCV | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| **Random Forest** | RandomizedSearchCV | `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf` |
| **Gradient Boosting** | RandomizedSearchCV | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_split` |
| **SVR** | GridSearchCV | `C`, `epsilon`, `gamma` |

**Why RandomizedSearchCV for RF and GB?**
These models have large hyperparameter spaces. RandomizedSearchCV samples `n_iter=20` combinations randomly, making it far faster than exhaustive grid search while still finding near-optimal configurations.

**Before vs. After Tuning** bar chart is included to visualize improvement.

---

### Section 5 — Best Model Selection

Final model selection is based on:
1. Highest R² on the hold-out test set
2. Lowest MSE and MAE
3. Best mean CV R² (generalizes well across folds)
4. Residuals approximately centered at zero (well-calibrated)

**Additional analysis included:**
- Feature importance bar chart (for tree-based models)
- Residual vs. Predicted scatter plot
- Residual distribution histogram

---

## 📈 Results Summary

### Base Model Performance (Before Tuning)

| Model | MSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~0.526 | ~0.533 | ~0.606 |
| Decision Tree | ~0.494 | ~0.454 | ~0.630 |
| SVR | ~0.357 | ~0.398 | ~0.728 |
| Random Forest | ~0.255 | ~0.328 | ~0.806 |
| **Gradient Boosting** | **~0.294** | **~0.371** | **~0.776** |

### After Hyperparameter Tuning

| Model | R² (Tuned) | Improvement |
|---|---|---|
| Ridge Regression | ~0.61 | +0.01 |
| Decision Tree | ~0.72 | +0.09 |
| SVR | ~0.76 | +0.03 |
| Random Forest | ~0.83 | +0.02 |
| **Gradient Boosting** | **~0.84** | **+0.06** |

> ℹ️ Exact values may vary slightly based on sklearn version and system. `random_state=42` is fixed throughout for reproducibility.

---

## 🏆 Best Model — Gradient Boosting Regressor

**Why Gradient Boosting wins:**

| Factor | Explanation |
|---|---|
| **Non-linearity** | Captures complex relationships between income, geography, and house price |
| **Sequential correction** | Each tree targets residual errors of the previous ensemble, steadily improving accuracy |
| **Regularization** | `learning_rate` + `subsample` prevent overfitting |
| **Feature interactions** | Automatically learns interactions (e.g., income × location) |
| **Tuning payoff** | Most responsive to hyperparameter tuning among all models |

**Most important features** (by Gradient Boosting feature importances):
1. `MedInc` — Median income is the strongest predictor of house value
2. `Latitude` / `Longitude` — Geography captures coastal vs. inland price differences
3. `HouseAge` — Older neighborhoods in prime locations retain value

---

## ⚠️ Worst Model — Decision Tree (Untuned)

An untuned Decision Tree grows to full depth, memorizing the training set but generalizing poorly. After tuning (`max_depth` constrained), performance improves substantially — demonstrating how critical regularization is for tree-based models.

---

## 🔑 Key Takeaways

- **Data leakage prevention** is critical — always fit scalers/transformers on training data only
- **Ensemble methods** (Random Forest, Gradient Boosting) consistently outperform single models on tabular data
- **Cross-validation** gives a more reliable performance estimate than a single train/test split
- **Hyperparameter tuning** can yield meaningful gains, especially for Gradient Boosting and Decision Trees
- **Feature importance** analysis confirms that domain knowledge aligns with model findings (income → price)

---

## 👤 Muhsina v s

**Assignment:** ML Assignment 4 — Supervised Learning: Regression  
**Course:** Machine Learning  
**Dataset:** California Housing (UCI / sklearn)

---

## 📄 License : MIT Licence

This project is submitted as part of an academic assignment. Code is open for educational reference.
