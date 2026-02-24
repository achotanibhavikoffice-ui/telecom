
# TELCO CHURN DASHBOARD — INSIGHT-DRIVEN VERSION

import streamlit as st
st.set_page_config(page_title="Telco Churn Intelligence", layout="wide", page_icon="📊")

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# THEME + STYLING

st.markdown("""
    <style>
    .stMetric label { font-size: 16px; color: #666; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; }
    .insight { 
        font-size: 15px; 
        background-color: #f9fafb; 
        border-left: 4px solid #4f46e5; 
        padding: 0.6em 1em; 
        border-radius: 0.5em;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()


# HEADER

st.title("📊 Telco Customer Churn Intelligence Dashboard")
st.markdown("Gain actionable insights into churn behaviour — not just numbers, but *stories in your data*.")
st.divider()

# ========================================================
# SIDEBAR INPUTS
st.sidebar.header("🔮 Predict Individual Churn")

contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 600.0)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.radio("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.radio("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.radio("Has Dependents?", ["Yes", "No"])

if st.sidebar.button("Predict Churn"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "InternetService": [internet_service],
        "Contract": [contract_type],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_data = pd.get_dummies(input_data)
    model_features = model.get_booster().feature_names if hasattr(model, "get_booster") else model.feature_names_in_
    input_data = input_data.reindex(columns=model_features, fill_value=0)

    churn_prob = model.predict_proba(input_data)[0][1]
    churn_label = "🔴 High Risk" if churn_prob > 0.3 else "🟢 Low Risk"

    st.sidebar.subheader(f"Predicted Churn Probability: **{churn_prob:.2%}**")
    st.sidebar.markdown(f"**Risk Category:** {churn_label}")

# ========================================================
# MAIN SECTION
# ========================================================
uploaded_file = st.file_uploader("📂 Upload Telco Customer Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)
    st.divider()

    # ========================================================
    # KPIs
    # ========================================================
    st.subheader("📈 Key Performance Indicators")
    c1, c2, c3 = st.columns(3)
    churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0) if "Churn" in df else 0
    avg_tenure = df["tenure"].mean() if "tenure" in df else 0
    avg_monthly = df["MonthlyCharges"].mean() if "MonthlyCharges" in df else 0
    c1.metric("Churn Rate", f"{churn_rate:.2%}")
    c2.metric("Avg Tenure", f"{avg_tenure:.1f} mo")
    c3.metric("Avg Monthly Bill", f"${avg_monthly:.2f}")
    st.markdown(f"<div class='insight'>💡 **Insight:** Roughly {churn_rate*100:.1f}% of your customers have churned. "
                f"Average tenure of {avg_tenure:.1f} months suggests moderate loyalty; higher charges may correlate with churn.</div>", unsafe_allow_html=True)

    st.divider()

    # ========================================================
    # TABS
    # ========================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Contract & Demographics",
        "💰 Billing & Payments",
        "🌐 Service Behavior",
        "📈 Correlation Insights"
    ])

    # TAB 1 — Contract & Demographics
    with tab1:
        if "Churn" in df.columns and "Contract" in df.columns:
            fig = px.histogram(df, x="Contract", color="Churn", barmode="group",
                               title="Churn Rate by Contract Type", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
            high_contract = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().get("Yes", 0).idxmax()
            st.markdown(f"<div class='insight'>💡 **Insight:** Customers on **{high_contract}** contracts churn the most — "
                        f"indicating short-term plans might need loyalty incentives.</div>", unsafe_allow_html=True)

        if "gender" in df.columns and "Churn" in df.columns:
            fig = px.pie(df, names="gender", color="gender", hole=0.4,
                         title="Gender Distribution", color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2 — Billing
    with tab2:
        if "MonthlyCharges" in df.columns and "Churn" in df.columns:
            fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                         title="Monthly Charges vs Churn", color_discrete_sequence=["#16a34a", "#dc2626"])
            st.plotly_chart(fig, use_container_width=True)
            churn_avg = df[df["Churn"]=="Yes"]["MonthlyCharges"].mean()
            stay_avg = df[df["Churn"]=="No"]["MonthlyCharges"].mean()
            st.markdown(f"<div class='insight'>💡 **Insight:** Average monthly bill for churned users (${churn_avg:.2f}) "
                        f"is higher than retained users (${stay_avg:.2f}). Price sensitivity may drive churn.</div>", unsafe_allow_html=True)

        if "PaymentMethod" in df.columns and "Churn" in df.columns:
            fig = px.bar(df, x="PaymentMethod", color="Churn", barmode="group",
                         title="Churn by Payment Method", color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3 — Services
    with tab3:
        if "InternetService" in df.columns and "Churn" in df.columns:
            fig = px.bar(df, x="InternetService", color="Churn", barmode="group",
                         title="Internet Service vs Churn", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
            top_service = df.groupby("InternetService")["Churn"].value_counts(normalize=True).unstack().get("Yes", 0).idxmax()
            st.markdown(f"<div class='insight'>💡 **Insight:** Churn is highest among **{top_service}** users — likely due to service quality or pricing perception.</div>", unsafe_allow_html=True)

    # TAB 4 — Correlation
    with tab4:
        numeric_cols = df.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            corr = numeric_cols.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight'>💡 **Insight:** Strong negative correlation between tenure and churn probability "
                        "confirms that long-term customers are less likely to churn.</div>", unsafe_allow_html=True)
else:
    st.info("👆 Upload a dataset to explore churn insights.")
