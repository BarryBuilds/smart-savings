                    "Recommended action: Consider follow-up "
                    "communication or another savings product."
                )

    # ================ TAB 2: SAVER GROUP PREDICTOR ================
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

    # ================ TAB 3: ABOUT THE MODELS ================
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
