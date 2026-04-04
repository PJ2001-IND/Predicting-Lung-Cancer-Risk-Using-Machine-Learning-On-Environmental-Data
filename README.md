# 🫁 Predicting Lung Cancer Risk Using Machine Learning on Environmental Data

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square&logo=scikit-learn)

> A machine learning project that predicts the risk level of lung cancer in patients based on environmental exposure factors, lifestyle habits, and clinical symptoms — enabling early-stage risk stratification.

---

## 📌 Problem Statement

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is critical for improving patient outcomes, yet diagnosis often comes too late. This project explores whether environmental and lifestyle data — such as air quality exposure, smoking habits, and occupational hazards — can be used to predict a patient's lung cancer risk level using supervised machine learning.

---

## 🎯 Objective

- Build and compare multiple ML classification models to predict lung cancer risk (Low / Medium / High)
- Identify the most influential environmental and lifestyle features driving lung cancer risk
- Provide a replicable, data-driven pipeline for medical risk stratification

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `cancer patient data sets.csv` |
| Source | Publicly available cancer patient dataset |
| Target Variable | `Level` — Risk classification (Low, Medium, High) |
| Features | 24 patient attributes covering environmental, lifestyle, and symptom factors |

### Key Features Include:
- **Environmental**: Air Pollution, Dust Allergy, Occupational Hazards, passive smoking
- **Lifestyle**: Smoking, Alcohol Use, Balanced Diet, Obesity
- **Symptoms**: Chest Pain, Coughing of Blood, Fatigue, Shortness of Breath, Wheezing
- **Clinical**: Age, Gender, Genetic Risk, Chronic Lung Disease

---

## 🔬 Methodology

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │   ├── Distribution plots
   │   ├── Correlation heatmap
   │   └── Feature importance analysis
   │
   ▼
Data Preprocessing
   │   ├── Label encoding (target variable)
   │   ├── Feature scaling
   │   └── Train-test split (80/20)
   │
   ▼
Model Training & Comparison
   │   ├── Logistic Regression
   │   ├── Decision Tree
   │   ├── Random Forest
   │   └── (Additional classifiers)
   │
   ▼
Evaluation
       ├── Accuracy Score
       ├── Confusion Matrix
       ├── Classification Report (Precision, Recall, F1)
       └── Feature Importance Plot
```

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | — |
| Decision Tree | — |
| Random Forest | — |

> 📝 *Refer to the notebook `LungCancerPrediction.ipynb` for full results, confusion matrices, and visualizations.*

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | ML model building and evaluation |
| Jupyter Notebook | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Predicting-Lung-Cancer-Risk-Using-Machine-Learning-On-Environmental-Data.git

# Navigate to the project directory
cd Predicting-Lung-Cancer-Risk-Using-Machine-Learning-On-Environmental-Data

# Launch Jupyter Notebook
jupyter notebook LungCancerPrediction.ipynb
```

---

## 📁 Project Structure

```
📦 Predicting-Lung-Cancer-Risk-Using-ML
 ┣ 📓 LungCancerPrediction.ipynb   # Main analysis and model notebook
 ┣ 📄 cancer patient data sets.csv  # Dataset
 ┗ 📄 README.md                     # Project documentation
```

---

## 💡 Key Insights

- Environmental factors like **air pollution** and **occupational hazards** are among the strongest predictors of high lung cancer risk
- Lifestyle features such as **smoking** and **alcohol use** compound environmental risks significantly
- Tree-based ensemble models tend to outperform linear models on this dataset due to non-linear feature interactions

---

## 🔭 Future Scope

- Deploy as a web application using Streamlit for real-time risk prediction
- Incorporate SHAP values for deeper model explainability
- Extend dataset with real-world clinical records for better generalization
- Experiment with XGBoost and Neural Networks for performance improvement

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

> ⭐ If you found this project useful, consider giving it a star!
