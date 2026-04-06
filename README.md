# 🫁 Predicting Lung Cancer Risk Using Machine Learning on Environmental Data

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square&logo=scikit-learn)
![AI Powered](https://img.shields.io/badge/AI-Powered-purple?style=flat-square&logo=openai)
![Status](https://img.shields.io/badge/Status-Production--Grade%20In%20Progress-orange?style=flat-square)

> A machine learning research project developed as part of a Final Thesis Report that predicts lung cancer risk in patients using environmental exposure factors, lifestyle habits, demographic data, and clinical symptoms — enabling early-stage risk stratification without reliance on medical imaging or laboratory diagnostics. This project is actively being evolved into a **production-grade, AI-powered medical decision support tool**.

---

## 📌 Problem Statement

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is critical for improving patient outcomes, yet diagnosis often comes too late due to heavy dependence on costly imaging and clinical investigations. This project explores whether environmental and lifestyle data — such as air quality exposure, smoking habits, and occupational hazards — can be used to predict a patient's lung cancer risk using supervised machine learning, providing a scalable and accessible screening tool especially suited for low-resource settings.

---

## 🎯 Objectives

- Build and compare multiple ML classification models to predict lung cancer risk (Low / Medium / High)
- Identify the most influential environmental, behavioral, and demographic features driving lung cancer risk
- Demonstrate the feasibility of integrating lifestyle and environmental factors into predictive modeling
- Provide a replicable, data-driven ML pipeline for medical risk stratification that supports public health decision-making
- **Evolve the research prototype into a production-grade, AI-powered medical screening tool** accessible to clinicians and public health professionals worldwide

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `cancer patient data sets.csv` |
| Source | [Kaggle — Cancer Patients and Air Pollution: A New Link](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link?resource=download) |
| Records | 1,000 patient entries |
| Target Variable | `Level` — Risk classification: Low (303), Medium (332), High (365) |
| Total Features | 25 patient attributes covering environmental, lifestyle, symptom, and demographic factors |
| Age Range | 14 – 73 years |

### Feature Categories

| Category | Features |
|---|---|
| **Environmental** | Air Pollution, Dust Allergy, Occupational Hazards, Passive Smoker |
| **Lifestyle** | Smoking, Alcohol Use, Balanced Diet, Obesity |
| **Symptoms** | Chest Pain, Coughing of Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring |
| **Demographic / Clinical** | Age, Gender, Genetic Risk, Chronic Lung Disease |

---

## 🔬 Methodology

```
Raw Dataset (1,000 records, 25 features)
   │
   ▼
Exploratory Data Analysis (EDA)
   │   ├── Univariate analysis & distribution plots
   │   ├── Correlation heatmap
   │   ├── Chi-square statistical tests (categorical features)
   │   └── Outlier detection via Z-Score
   │
   ▼
Data Preprocessing
   │   ├── Drop non-predictive columns (Patient Id)
   │   ├── Binary encoding of target variable (Low → 0, Medium/High → 1)
   │   ├── Feature scaling (StandardScaler)
   │   ├── Categorical encoding (OneHotEncoder)
   │   └── Train-test split (70/20) with stratification
   │
   ▼
Feature Selection
   │   ├── Statistical significance (Chi-square)
   │   ├── Correlation with lung cancer outcome
   │   ├── Clinical & epidemiological relevance
   │   └── Removal of redundant features
   │   Selected: Age, Gender, Smoking, Alcohol Use, Air Pollution
   │
   ▼
Model Training & Evaluation (Scikit-learn Pipelines)
   │   ├── Logistic Regression (baseline, max_iter=1000)
   │   ├── Decision Tree Classifier (random_state=42)
   │   └── Random Forest Classifier (n_estimators=200, class_weight='balanced')
   │
   ▼
Evaluation & Interpretation
       ├── Accuracy, Precision, Recall, F1-Score, ROC-AUC
       ├── Confusion Matrices
       ├── ROC Curves
       ├── 5-Fold Cross Validation
       ├── Feature Importance (Random Forest — Gini)
       └── Logistic Regression Coefficients
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Decision Tree | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **Random Forest** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |

> 📝 *All three models achieved perfect predictive performance on this dataset. The Random Forest model demonstrated the highest robustness, lowest false negative rate, and best cross-validation stability — making it the recommended model for deployment. Refer to `LungCancerPrediction.ipynb` for full confusion matrices, ROC curves, and visualizations.*

---

## 💡 Key Findings

- **Smoking behavior** emerged as the strongest single predictor of lung cancer risk across all models
- **Air pollution exposure** was the most influential environmental predictor, validating the inclusion of environmental data in predictive modeling
- **Age** was a dominant demographic predictor, reflecting the cumulative nature of long-term environmental and behavioral exposure
- **Alcohol use** showed a weaker but observable contribution compared to smoking and pollution
- **Gender** had a limited but present contribution, suggesting behavioral factors mediate demographic differences
- **Random Forest** outperformed Logistic Regression and Decision Tree in robustness and cross-validation stability due to its ensemble structure reducing variance
- The framework demonstrates that **non-clinical, non-imaging data** can power accurate lung cancer risk prediction, enabling accessible public health screening

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | Data visualization (distributions, heatmaps, ROC curves) |
| SciPy | Statistical testing (Chi-square, Z-Score outlier detection) |
| Scikit-learn | ML pipeline building, model training, and evaluation |
| Jupyter Notebook | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Predicting-Lung-Cancer-Risk-Using-Machine-Learning-On-Environmental-Data.git

# Navigate to the project directory
cd Predicting-Lung-Cancer-Risk-Using-Machine-Learning-On-Environmental-Data

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook LungCancerPrediction.ipynb
```

> ⚠️ **Note:** Ensure the dataset file `cancer patient data sets.csv` is placed in the same directory as the notebook before running.

---

## 📁 Project Structure

```
📦 Predicting-Lung-Cancer-Risk-Using-ML
 ┣ 📓 LungCancerPrediction.ipynb        # Main ML pipeline and analysis notebook
 ┣ 📄 cancer patient data sets.csv      # Dataset (1,000 records, 25 features)
 ┣ 📄 requirements.txt                  # Python dependencies
 ┗ 📄 README.md                         # Project documentation
```

---

## 🏥 Vision — Production-Grade Medical AI Tool

This project is being actively developed beyond its research origins into a **full-scale, production-grade medical decision support system**. The goal is to transform this ML pipeline into a clinically deployable AI tool that can assist healthcare professionals, public health bodies, and individuals in identifying lung cancer risk early — without requiring expensive imaging or laboratory tests.

### 🤖 Planned AI Integration

Aligned with the research findings from the thesis, the following AI capabilities are planned for integration:

| AI Capability | Purpose | Technology |
|---|---|---|
| **Explainable AI (XAI)** | Generate patient-level explanations for every prediction — *why* a patient is high risk | SHAP, LIME |
| **Large Language Model (LLM) Layer** | Natural language risk reports — clinician-readable summaries generated from model output | GPT-4 / Claude API |
| **Conversational Risk Assessment** | AI chatbot interface for patients to input their lifestyle data through conversation | LangChain + LLM |
| **AutoML & Model Self-Improvement** | Automated retraining pipelines when new data is available | AutoSklearn, MLflow |
| **Temporal Risk Modeling** | Track a patient's risk trajectory over time using longitudinal data | LSTM / Time-series ML |
| **Federated Learning** | Train on distributed hospital data without centralising sensitive patient records | PySyft / Flower |
| **Real-Time Air Quality Integration** | Pull live environmental pollution data to enrich predictions dynamically | OpenAQ API / WHO Data |

### 🏗️ Production Architecture Roadmap

```
Patient / Clinician Input (Web / Mobile / API)
   │
   ▼
AI Conversational Interface (LLM-powered chatbot)
   │
   ▼
Data Validation & Preprocessing Layer
   │
   ▼
Core ML Prediction Engine (Random Forest + XGBoost ensemble)
   │
   ▼
Explainability Engine (SHAP — feature-level patient report)
   │
   ▼
LLM Report Generator (plain-language clinical risk summary)
   │
   ▼
Risk Dashboard (clinician-facing) / Alert System (public health)
   │
   ▼
Audit Log & Compliance Layer (HIPAA / GDPR ready)
```

### 🔭 Development Roadmap

**Phase 1 — Enhanced ML (In Progress)**
- Integrate XGBoost and ensemble stacking for improved generalization
- Add SHAP explainability for every individual prediction
- External validation using independent datasets

**Phase 2 — AI & NLP Layer**
- LLM-generated clinical risk reports in plain English
- Conversational patient intake chatbot (LangChain + LLM)
- Automated alert system for high-risk patient flagging

**Phase 3 — Production Deployment**
- REST API (FastAPI) with secure authentication
- Streamlit / React web dashboard for clinicians
- Docker containerization and cloud deployment (AWS / GCP)
- HIPAA/GDPR-compliant data handling and audit logging

**Phase 4 — Scale & Generalization**
- Federated learning for multi-hospital training without data sharing
- Real-time environmental data (air quality APIs) integration
- Multi-regional dataset expansion for global applicability
- Mobile application for community-level public health screening

---

## 📚 Research Context

This project was developed as a **Final Thesis Report** (February 2026) exploring the use of machine learning for lung cancer risk stratification using non-clinical, non-imaging data. The methodology was designed to produce a scalable and accessible prediction framework suitable for public health settings, particularly in low-resource environments where advanced diagnostic infrastructure may be unavailable.

**Notebook on Google Drive:** [LungCancerPrediction.ipynb](https://drive.google.com/drive/folders/14-w4XGmku0c6i79rt8fiHkK-bOd0zmXC?usp=sharing)

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

> ⭐ If you found this project useful, consider giving it a star!
