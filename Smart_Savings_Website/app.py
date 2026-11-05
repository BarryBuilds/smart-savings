import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Savings",
    page_icon="💰",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""

<style>
.stApp {
    background-color: #f4f7fb;
    color: #263445;
}
/* Improve text visibility */
.stApp {
    color: #17365d;
}

.stApp p,
.stApp label,
.stApp span,
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #17365d;
}

.stTabs [data-baseweb="tab"] {
    color: #17365d !important;
}

.stTabs [aria-selected="true"] {
    color: #2874a6 !important;
}
h1, h2, h3, p, label,
.stMarkdown, .stCaption {
    color: #263445;
}

.hero {
    background: linear-gradient(120deg, #17365d, #2874a6);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h1, .hero p {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e8f0;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #17365d !important;
}

.stTabs [data-baseweb="tab"] {
    color: #17365d !important;
}

.stTabs [aria-selected="true"] {
    color: #2874a6 !important;
}

.stButton button,
.stFormSubmitButton button {
    color: white;
    background-color: #17365d;
    border: none;
}

</style>



BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    savings = joblib.load(BASE_DIR / "savings_model.joblib")
    saver = joblib.load(BASE_DIR / "saver_model.joblib")
    customers = pd.read_csv(BASE_DIR / "customers_with_groups.csv")
    return savings, saver, customers

try:
    savings, saver, customers = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Could not load the models: {e}")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>💰 Smart Savings</h1>
    <p>Your intelligent assistant for smarter saving decisions.</p>
</div>
""", unsafe_allow_html=True)

if models_loaded:

    st.caption("Machine learning demonstration • Educational use only")

    deposit_tab, group_tab, about_tab = st.tabs([
        "🏦 Fixed Deposit Predictor",
        "👥 Saver Group Predictor",
        "📊 About the Models"
    ])

    # ================ TAB 1 ================
    with deposit_tab:
        st.header("Fixed Deposit Prediction")
        st.write(
            "Enter customer information to estimate the chance "
            "of opening a fixed deposit."
        )

        with st.form("deposit_form"):

            col1, col2 = st.columns(2)

            with col1:
                age = st.slider("Customer age", 17, 98, 40)

                job = st.selectbox(
                    "Job",
                    savings["choices"]["job"]
                )

                marital = st.selectbox(
                    "Marital status",
                    savings["choices"]["marital"]
                )

                education = st.selectbox(
                    "Education",
                    savings["choices"]["education"]
                )

                default = st.selectbox(
                    "Credit default",
                    savings["choices"]["default"]
                )

            with col2:
                housing = st.selectbox(
                    "Housing loan",
                    savings["choices"]["housing"]
                )

                loan = st.selectbox(
                    "Personal loan",
                    savings["choices"]["loan"]
                )

                campaign = st.slider(
                    "Calls in this campaign", 1, 20, 2
                )

                previous = st.slider(
                    "Calls in previous campaigns", 0, 7, 0
                )

                poutcome = st.selectbox(
                    "Previous campaign result",
                    savings["choices"]["poutcome"]
                )

                euribor3m = st.slider(
                    "Market interest rate (Euribor 3 months)",
                    0.6, 5.1, 4.9, 0.1
                )

            submitted = st.form_submit_button(
                "Predict Customer Chance",
                use_container_width=True
            )

        if submitted:
            customer = pd.DataFrame([{
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "campaign": campaign,
                "previous": previous,
                "poutcome": poutcome,
                "euribor3m": euribor3m
            }])

            chance = savings["model"].predict_proba(
                customer[savings["features"]]
            )[0, 1]

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                "Chance of opening a fixed deposit",
                f"{chance * 100:.1f}%"
            )

            st.progress(float(chance))

            if chance >= 0.5:
                st.success(
                    "Recommended action: Prioritize this customer "
                    "for a fixed deposit offer."
                )
            else:
                st.info(
                    "Recommended action: Consider follow-up "
                    "communication or another savings product."
                )

    # ================ TAB 2 ================
    with group_tab:
        st.header("Discover Your Saver Group")
        st.write(
            "Enter annual income and spending score to discover "
            "your customer group."
        )

        with st.form("group_form"):
            col1, col2 = st.columns(2)

            with col1:
                income = st.number_input(
                    "Annual income (k$)",
                    min_value=0.0,
                    max_value=200.0,
                    value=90.0,
                    step=1.0
                )

            with col2:
                spending = st.slider(
                    "Spending score",
                    1, 100, 20
                )

            find_group = st.form_submit_button(
                "Find My Saver Group",
                use_container_width=True
            )

        if find_group:
            person = pd.DataFrame([{
                "Annual Income (k$)": income,
                "Spending Score (1-100)": spending
            }])

            group = int(
                saver["model"].predict(
                    person[saver["features"]]
                )[0]
            )

            names = {
                0: "Balanced Savers",
                1: "Premium Spenders",
                2: "Enthusiastic Spenders",
                3: "Wealthy Cautious Savers",
                4: "Budget-Conscious Savers"
            }

            tips = {
                0: "Create a monthly budget and save a fixed amount regularly.",
                1: "Pay yourself first before spending on luxury purchases.",
                2: "Track spending and set a realistic weekly savings target.",
                3: "Build an emergency fund and explore fixed deposits.",
                4: "Start with small, regular savings amounts."
            }

            st.divider()
            st.subheader("Your Customer Profile")

            st.metric("Predicted Group", f"Group {group}")
            st.success(names[group])

            st.markdown("### 💡 Personalized Savings Advice")
            st.info(tips[group])

    # ================ TAB 3 ================
    with about_tab:
        st.header("About the Machine Learning Models")

        st.subheader("1. Supervised Learning")
        st.write(
            "Random Forest predicts the probability that a bank "
            "customer will open a fixed deposit."
        )

        st.subheader("2. Unsupervised Learning")
        st.write(
            "K-Means clustering identifies five customer groups "
            "based on annual income and spending score."
        )

        st.subheader("Model Evaluation")

        scores = pd.DataFrame(savings["scores"])
        st.dataframe(scores, use_container_width=True)

        st.write(
            "**Why F1-score?** The dataset is imbalanced, with "
            "far fewer customers opening fixed deposits. F1 "
            "balances precision and recall, so it is more useful "
            "than accuracy alone for selecting the model."
        )

        st.subheader("Customer Dataset")
        st.write(f"Total customer records: {len(customers)}")

        st.caption(
            "Dataset used for educational purposes only. "
            "Predictions are not financial advice."
        )
