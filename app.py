import streamlit as st
import pandas as pd
from predictor import ProfessionalNHLPredictor

st.set_page_config(page_title="NHL Betting Predictor", page_icon="🏒", layout="wide")

st.title("🏒 Pro NHL Betting Predictor")
st.caption("This application uses an advanced **XGBoost** Machine Learning model to predict NHL games and find **Positive Expected Value (+EV)** bets based on real-time data from the NHL OpenAPI and MoneyPuck.")

@st.cache_resource
def load_predictor():
    app = ProfessionalNHLPredictor()
    app.train_real_model()
    return app

predictor = load_predictor()

st.header("Today's NHL Predictions")

if st.button("Refresh Live Odds & Predictions"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)  # Cache results for 1 hour to avoid spamming the APIs
def get_daily_predictions_v3():
    return predictor.run_daily_predictions()

with st.spinner("Fetching live NHL stats, training models, and pulling odds..."):
    results = get_daily_predictions_v3()
    
if not results:
    st.info("No NHL games are scheduled for today, or data could not be retrieved.")
else:
    for res in results:
        with st.container():
            exact_sc = res.get('exact_score', 'Data Loading...')
            st.markdown(f"### {res['matchup']}   |   🎯 Pred. Score: **{exact_sc}**")
            
            # --- Moneyline UI ---
            st.markdown("#### Moneyline")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Winner", res['predicted_winner'])
            with col2:
                st.metric("Model Confidence", res['confidence'])
            with col3:
                st.metric("Live ML Odds", res['odds'])
            
            ev_ml = res['ev']
            if ev_ml > 0:
                st.success(f"📈 **Moneyline Edge Detected!** Expected Value: +${ev_ml:.2f} per $100 bet")
            else:
                st.warning(f"📉 **No ML Edge.** Expected Value: ${ev_ml:.2f} per $100 bet. Skip.")
                
            # --- Over/Under Totals UI ---
            st.markdown("#### Over/Under (Totals)")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Vegas O/U Line", f"{res['o_u_line']}")
            with col5:
                st.metric("Proj. Total Goals", f"{res['projected_total']}")
            with col6:
                st.metric("Live O/U Odds", f"O {res['over_odds']} // U {res['under_odds']}")
            
            ev_over = res['ev_over']
            ev_under = res['ev_under']
            if ev_over > 0:
                st.success(f"📈 **OVER Edge Detected!** Expected Value: +${ev_over:.2f} per $100 bet")
            elif ev_under > 0:
                st.success(f"📈 **UNDER Edge Detected!** Expected Value: +${ev_under:.2f} per $100 bet")
            else:
                st.warning(f"📉 **No O/U Edge.** (Over EV: ${ev_over:.2f} | Under EV: ${ev_under:.2f}). Skip.")
            
            st.caption(f"Data Source: {res.get('data_source', 'Odds API')}")
            st.divider()

st.sidebar.title("About the Model")
st.sidebar.info("""
**Architecture**:
- **Data**: Live NHL standings & ultra-deep MoneyPuck analytics (Corsi, Fenwick, High-Danger xG)
- **Machine Learning**: XGBoost hyperparameter-tuned on 6,700 real-world games
- **Prediction**: Moneyline value (+EV), Exact Score Simulator, and Poisson Over/Under Distribution
- **Odds**: Connected to The Odds API via API key, or mathematically mocked.
""")
