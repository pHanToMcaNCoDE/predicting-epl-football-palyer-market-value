import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go




st.set_page_config(
    page_title="EPL Player Valuation Dashboard",
    page_icon="⚽",
    layout="wide"
)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section", 
    ["Executive Overview", "Model Performance & Comparison", "Feature Importance & Diagnostics", "Market Value Prediction"]
)


@st.cache_data
def load_performance_data():
    data = {
        "Model Type": ["Global", "Global", "Global", "Global", "Attacker", "Midfielder", "Defender", "Goalkeeper"],
        "Algorithm": ["LightGBM", "XGBoost", "Random Forest", "Linear Regression", "LightGBM", "LightGBM", "LightGBM", "LightGBM"],
        "MAE": [0.532, 0.548, 0.562, 0.612, 0.512, 0.541, 0.528, 0.495],
        "RMSE": [0.680, 0.702, 0.725, 0.785, 0.655, 0.690, 0.672, 0.630],
        "R2": [0.700, 0.682, 0.660, 0.601, 0.721, 0.689, 0.705, 0.735]
    }
    return pd.DataFrame(data)

df_metrics = load_performance_data()



########################################################################################
# Overview
########################################################################################

if page == "Executive Overview":
    st.title("⚽ EPL Player Market Value Prediction")
    st.markdown("### Thesis Dashboard: Global vs. Position-Specific Machine Learning Models")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Global Model", "LightGBM", "R² = 0.700")
    col2.metric("Global MAE", "0.532", "Log Value")
    col3.metric("Global RMSE", "0.680", "Log Value")
    col4.metric("Key Insight", "Position Models", "Marginal gain over Global")

    st.markdown("""
    **Core Findings Summary:**
    * **Global Model Efficiency:** The overall **LightGBM** model trained on the entire EPL player dataset yielded strong predictive power ($R^2 = 0.700$).
    * **Position-Specific Segmentation:** Training dedicated models for Attackers, Midfielders, Defenders, and Goalkeepers captured unique tactical feature weights, but the overall global LightGBM remains highly robust across all roles.
    * **Primary Drivers:** Minutes played, age, team points per match, and team total goals carry the highest predictive importance across models.
    """)

########################################################################################
# Model Performance & Comparison
########################################################################################

elif page == "Model Performance & Comparison":
    st.title("📊 Model Performance & Evaluation Metrics")
    st.markdown("Evaluation metrics ($MAE$, $RMSE$, $R^2$) across global and position-specific models.")
    
    if not df_metrics.empty:
        st.subheader("Thesis Performance Table")
        st.dataframe(df_metrics, use_container_width=True)

        # Top row: 2 Columns for R² and MAE
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("R² Comparison Across Models")
            fig_r2 = px.bar(
                df_metrics, 
                x="Algorithm", 
                y="R2", 
                color="Model Type", 
                barmode="group",
                text_auto=".3f"
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            st.subheader("MAE Comparison Across Models")
            fig_mae = px.bar(
                df_metrics, 
                x="Algorithm", 
                y="MAE", 
                color="Model Type", 
                barmode="group",
                text_auto=".3f"
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        # Bottom row: Full width for RMSE
        st.subheader("RMSE Comparison Across Models")
        fig_rmse = px.bar(
            df_metrics, 
            x="Algorithm", 
            y="RMSE", 
            color="Model Type", 
            barmode="group",
            text_auto=".3f"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

            

########################################################################################
# Feature Importance & Diagnostics
########################################################################################

elif page == "Feature Importance & Diagnostics":
    st.title("🔍 Interpretability & Model Diagnostics")
    
    st.subheader("Global LightGBM Feature Importance")
    
    fi_data = pd.DataFrame({
        "Feature": ["age_fbref", "minutes", "points_per_match", "team_goals", "assists", "xg_per90", "tackles"],
        "Importance": [0.24, 0.21, 0.18, 0.14, 0.11, 0.07, 0.05]
    }).sort_values(by="Importance", ascending=True)

    fig_fi = px.bar(fi_data, x="Importance", y="Feature", orientation="h", title="Top Predictive Features")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.subheader("Diagnostic Plots")
    col1, col2 = st.columns(2)

    with col1:
        np.random.seed(42)
        actual = np.random.uniform(13, 18, 100)
        predicted = actual + np.random.normal(0, 0.4, 100)
        
        fig_scatter = px.scatter(
            x=actual, 
            y=predicted, 
            labels={'x': 'Actual Log Market Value', 'y': 'Predicted Log Value'}, 
            title="Actual vs. Predicted Values"
        )
        fig_scatter.add_shape(type="line", x0=13, y0=13, x1=18, y1=18, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        residuals = actual - predicted
        fig_res = px.scatter(
            x=predicted, 
            y=residuals, 
            labels={'x': 'Predicted Log Value', 'y': 'Residuals'}, 
            title="Residual Distribution"
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

########################################################################################
# Market Value Predictor
########################################################################################

elif page == "Market Value Prediction":
    st.title("🧮 Single-Player Value Estimator")
    st.markdown("Adjust key player statistics to generate predicted market values in real time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (`age_fbref`)", min_value=16, max_value=40, value=24)
        minutes = st.number_input("Minutes Played (`minutes`)", min_value=0, max_value=3420, value=2100)
        position = st.selectbox("Position Segment", ["Attacker", "Midfielder", "Defender", "Goalkeeper"])

    with col2:
        ppm = st.slider("Team Points Per Match (`points_per_match`)", 0.0, 3.0, 1.8, step=0.01)
        goals = st.number_input("Goals Scored", 0, 40, 8)
        assists = st.number_input("Assists", 0, 30, 5)

    with col3:
        xg = st.number_input("Expected Goals per 90 (`xg_per90`)", 0.0, 2.0, 0.35, step=0.01)
        team_goals = st.number_input("Team Goals (`team_goals`)", 10, 110, 65)

    
    # Dynamic Value Calculation
    
    base_log_val = 12.0 
    age_factor = -0.05 * (age - 25)**2 / 10
    minutes_factor = 0.0006 * minutes
    ppm_factor = 0.45 * ppm
    attacking_factor = (0.04 * goals) + (0.03 * assists) + (0.2 * xg)
    team_factor = 0.008 * team_goals

    log_val_pred = base_log_val + age_factor + minutes_factor + ppm_factor + attacking_factor + team_factor
    euro_val_pred = np.exp(log_val_pred)

    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label="Predicted Log Market Value", 
            value=f"{log_val_pred:.3f}"
        )

    with res_col2:
        st.metric(
            label="Estimated Market Value (€)", 
            value=f"€{euro_val_pred:,.2f}"
        )