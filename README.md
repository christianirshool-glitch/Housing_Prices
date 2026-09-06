# 🏡 Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end data science project focused on predicting housing prices. The main goal was to build a reproducible ML pipeline and compare the performance of a linear model against a tree-based model, uncovering which approach best fits the underlying structure of the data.

---

## 📌 Table of Contents

- [Context](#-context)
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Tech Stack](#️-tech-stack)
- [Installation and Usage](#-installation-and-usage)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [License](#-license)

---

## 📌 Context

Housing prices depend on a mix of structural features (area, number of rooms, stories) and amenities (air conditioning, parking, preferred location). Understanding which of these factors actually drive price — and whether that relationship is linear or not — is key to choosing the right modeling approach and avoiding unnecessary complexity.

---

## 🎯 Objective

Build a model capable of predicting housing prices from their structural and categorical features, comparing a linear model against a tree-based ensemble to determine which one better captures the real relationship between features and price.

---

## 📊 Dataset

The dataset used is **`Housing.csv`** (source: [Kaggle — yasserh/housing-prices-dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)), containing **545 records** and **13 columns**:

| Column | Description |
|---|---|
| `price` | Target variable — house price |
| `area` | Plot/lot area |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `parking` | Number of parking spaces |
| `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | Binary (yes/no) categorical features |
| `furnishingstatus` | Furnishing status (furnished / semi-furnished / unfurnished) |

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)
Review of numerical and categorical variable distributions, outlier detection, and visualization to understand the relationship between each feature and price.

### 2. Encoding categorical features
- Binary (yes/no) features such as `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, and `prefarea` were mapped to `0`/`1`.
- The multi-class categorical feature `furnishingstatus` was transformed using **One-Hot Encoding**.

### 3. Data preprocessing
- **Robust Scaling**: `RobustScaler` was applied to numerical features to mitigate the impact of outliers in the housing market (extreme property sizes/prices). The scaler was **fit only on the training set** and then used to transform both the training and test sets, avoiding data leakage.

### 4. Association analysis
Feature importance was assessed using two complementary techniques depending on variable type:
- **Eta-squared (η²)** for categorical features (e.g. `mainroad`, `airconditioning`, `furnishingstatus`) against price.
- **Pearson correlation** for numerical features (e.g. `area`, `bathrooms`, `stories`) against price.

### 5. Modeling and evaluation
Three models were trained and compared:
- **Random Forest Regressor** (baseline).
- **Random Forest Regressor**, optimized via `GridSearchCV`.
- **Linear Regression**.

### 6. Error analysis
Prediction errors were reviewed across the price range to identify where the model performs worst.

---

## 📈 Key Results

| Model | Notes |
|---|---|
| Random Forest (baseline) | Underperformed relative to the linear model |
| Random Forest (GridSearchCV-optimized) | Improved over baseline, but still below Linear Regression |
| **Linear Regression** | **Best-performing model** |

**Best model — Linear Regression:**

| Metric | Value |
|---|---|
| R² | 0.644 |
| MAE | 925,543 |

- **Linear Regression outperformed both Random Forest variants**, including the one optimized via `GridSearchCV`. This indicates that the relationship between the available features and price is predominantly **linear**, and that the added complexity of a tree-based ensemble did not translate into better generalization on this dataset.
- **`area` is the single most influential feature**, accounting for **58%** of the predictive importance — by a wide margin, the strongest driver of price in this dataset.
- Ranked by relevance, the most influential features identified are: `area`, `bathrooms`, `stories`, `parking`, `bedrooms`, and `airconditioning`.
- **Practical takeaway**: when a dominant feature has a largely linear relationship with the target, a simpler model can outperform a more complex one — added model capacity isn't useful if there are no meaningful non-linear patterns for it to exploit.

---

## 🛠️ Tech Stack

| Library | Use |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, Random Forest, Linear Regression, GridSearchCV, metrics |
| `statsmodels` | Statistical analysis |
| `jupyter` | Interactive environment |

---

## 🚀 Installation and Usage

### Prerequisites
* All dependencies are listed in [requirements.txt](requirements.txt).
* Download `Housing.csv` from [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset), or configure Kaggle credentials for automatic download if the notebook uses `kagglehub`.

### Setup steps

```bash
# 1. Clone the repository
git clone https://github.com/christianirshool-glitch/Housing_Prices.git
cd Housing_Prices

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Launch the interactive environment
jupyter notebook "Project_2_Housing_Prices.ipynb"
```

---

## 📁 Project Structure

```
housing-price-prediction/
├── Project_2_Housing_Prices.ipynb   # Main notebook with the full pipeline
├── requirements.txt                  # Project dependencies
├── LICENSE                            # MIT license
└── README.md                          # Project documentation
```

---

## 👤 Author

**Christian Méndez Giraldo**
Data Scientist · MSc in Data Science
[GitHub](https://github.com/christianirshool-glitch)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
