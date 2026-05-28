
# Predicting Customer Churn in the Telecommunications Industry Using XGBoost with Explainable Dashboard Integration

An end-to-end predictive analytics and machine learning pipeline that identifies at-risk subscribers in the telecommunications sector. This repository pairs an optimized gradient-boosted ensemble model with an interactive Streamlit business intelligence dashboard to operationalize data-driven customer retention strategies.

## 📌 Project Overview
Customer churn—the voluntary termination of service subscriptions—poses significant financial challenges in the subscription economy. Recognizing that acquiring a new customer costs 5× to 25× more than retaining an existing one, this project delivers a proactive retention framework. 

By analyzing an analytical corpus of **2,000 anonymized subscribers**, the system processes 21 multi-modal features spanning customer demographics, account structures, billing habits, and network configurations to compute individual churn probabilities.

### 🚀 Key Features
* **Decoupled ML Pipeline:** Clean, reproducible pipeline covering raw data ingestion, feature alignment, and serialized inference using an advanced `XGBoost` classifier.
* **Feature Engineering Engine:** Constructs composite signals (e.g., active add-on service counts, encoded internet indicators) to condense sparse feature spaces into high-gain inputs.
* **Operational Streamlit Dashboard:** Renders real-time, interactive single-user inference panels alongside cohort segmentation tabs, box plots, and correlation heatmaps for non-technical stakeholders.
* **Risk Threshold Tuning:** Implements an adjustable decision threshold (optimized at `0.3`) to prioritize minority-class recall, intentionally catching high-risk customers before they churn.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.10+
* **Machine Learning Core:** XGBoost, Scikit-Learn
* **Data Engineering:** Pandas, NumPy
* **Deployment & Visualization:** Streamlit, Plotly Express
* **Model Serialization:** Joblib / Pickle

📁 Project Directory Structure
├── app.py                  # Core Streamlit Web Application & Dashboard Architecture
├── churn_model.pkl         # Serialized, trained XGBoost Classifier Model Binary
├── telco_large.csv         # Analytical Dataset Partition 1 (Records 1 - 1,000)
├── telco_large2.csv        # Analytical Dataset Partition 2 (Records 1,001 - 2,000)
└── README.md               # Repository Documentation

## 📊 Dataset Specifications & Features

The study combines two data partitions (`telco_large.csv` and `telco_large2.csv`) to create a comprehensive evaluation corpus of **2,000 records** across 5 distinct dimensions:

| Domain | Feature Name | Data Type | Description / Sample Values |
| :--- | :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical / Binary | Household composition, gender distribution, and age brackets. |
| **Account** | `tenure`, `Contract`, `PaymentMethod` | Continuous / Ordinal | Subscription age (0-71 months), billing frequency, and payment type. |
| **Billing** | `MonthlyCharges`, `TotalCharges` | Continuous Float | Financial commitment values ranging from $20.11 to $7,992.74. |
| **Network** | `InternetService`, `OnlineSecurity`, `TechSupport` | Nominal Multi-class | Core connection architecture (DSL, Fiber Optic) and add-on profiles. |
| **Target** | `Churn` | Binary Categorical | Final classification state (**Yes** = Churned, **No** = Retained). |

---

## 🔬 Modeling & Key Empirical Findings

### 1. Feature Importance Dominance
The underlying gradient-boosting tree splits reveal that customer risk is heavily concentrated within structural accounts rather than demographic classifications. Two primary features account for **65.8% of total predictive gain**:
* `Contract_Month-to-month` (41.5%): Indicating that flexible, contract-free parameters present zero switching costs for subscribers.
* `InternetService_Fiber optic` (24.3%): Highlighting a vital pricing sensitivity or delivery expectation mismatch at the premium technology tier.

### 2. Cohort Analytics
* **Tenure Differential:** Churned customers exhibit a significantly shorter mean tenure (**33.1 months**) compared to loyal subscribers (**37.6 months**).
* **Frictionless Payments:** Subscribers utilizing credit card auto-pay configurations demonstrate the absolute lowest churn rate (**16.8%**), establishing automated payment structures as effective administrative anchors.

---

## 💻 Installation & Local Deployment

Follow these steps to configure your local environment and spin up the interactive business intelligence dashboard:

### Prerequisite Setup
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/your-username/telco-churn-xgboost.git](https://github.com/your-username/telco-churn-xgboost.git)
   cd telco-churn-xgboost
```

2. Create and activate an isolated virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

```


3. Install the required dependencies:
```bash
pip install streamlit xgboost pandas numpy scikit-learn plotly

```



### Launching the Application

Run the Streamlit server from the root of your project directory:

```bash
streamlit run app.py

```

Once initialized, open your local web browser and navigate to the local network port (typically `http://localhost:8501`).

---

## 📈 Dashboard Interface Modules

The web deployment breaks into four distinct logical spaces designed to streamline executive decision-making:

1. **Individual Inference Panel:** A contextual sidebar allowing account managers to input custom customer configurations manually, instantly generating real-time churn risk classifications.
2. **Dynamic KPI Metrics:** Computes overall cohort churn distributions, billing means, and duration baselines directly from the integrated dataset.
3. **Contractual Segments:** Uses Plotly Express to visualize churn rates over contract limits, highlighting areas of high churn.
4. **Billing Trajectories:** Renders comparative distribution box plots mapping monthly expenditures against historical service retention.

---

## 🔮 Future Engineering Roadmaps

To scale the model's current evaluation limits, upcoming development iterations will focus on:

* **Advanced Class Rebalancing:** Integrating Synthetic Minority Over-sampling Techniques (`SMOTE`) into the preprocessing pipeline to systematically improve minority-class model recall.
* **Explainable AI Frameworks:** Embedding live `SHAP` (SHapley Additive exPlanations) visualizers directly inside the Streamlit client layout to deliver transparent, feature-level rationale for every inference generated.
* **Longitudinal Engineering:** Factoring in temporal metrics, customer service ticket frequencies, and micro-usage trends to capture dynamic behavioral changes over time.

---

## 🎓 Academic Affiliation & Mentorship

This project was designed and built as a core research initiative within the **Department of Information Technology, Thadomal Shahani Engineering College (TSEC), University of Mumbai**, for the Academic Year 2025–2026.

* **Principal Investigator:** Bhavik Achotani (`achotanibhavikoffice@gmail.com`)
* **Academic Supervisor:** Dr. Mukesh Israni

---

Developed with 💻, 📊, and Python. Open-sourced under the MIT License.

```

```
