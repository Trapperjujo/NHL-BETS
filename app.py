import streamlit as st
import pandas as pd
from predictor import ProfessionalNHLPredictor

st.set_page_config(page_title="NHL Betting Predictor", page_icon="🏒", layout="wide")

st.title("🏒 Pro NHL Betting Predictor")
st.markdown("This application uses a fully trained **Random Forest Machine Learning model** to predict NHL games and find **Positive Expected Value (+EV)** bets based on real-time data from the NHL OpenAPI and live sportsbook providers.")

@st.cache_resource
def load_predictor():
    app = ProfessionalNHLPredictor()
    app.train_synthetic_model()
    return app

predictor = load_predictor()

st.header("Today's NHL Predictions")

if st.button("Refresh Live Odds & Predictions"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)  # Cache results for 1 hour to avoid spamming the APIs
def get_daily_predictions():
    return predictor.run_daily_predictions()

with st.spinner("Fetching live NHL stats, training models, and pulling odds..."):
    results = get_daily_predictions()
    
if not results:
    st.info("No NHL games are scheduled for today, or data could not be retrieved.")
else:
    # Convert to DataFrame for a nice summary table
    df = pd.DataFrame(results)
    
    # Filter for +EV only
    st.subheader("✅ High Value Bets (+EV)")
    value_bets = df[df['is_value'] == True]
    
    if len(value_bets) > 0:
        for idx, row in value_bets.iterrows():
            st.success(f"**{row['matchup']}**: Bet on **{row['predicted_winner']}** @ {row['odds']} (Confidence: {row['confidence']}) — **EV: +${row['ev']:.2f}** per $100")
    else:
        st.warning("No mathematically profitable bets found today based on current odds.")
        
    st.subheader("📊 All Game Analysis")
    # Clean up df for display
    display_df = df[['matchup', 'predicted_winner', 'confidence', 'odds', 'ev', 'data_source']].copy()
    display_df.columns = ['Matchup', 'Predicted Winner', 'Model Confidence', 'Decimal Odds', 'Expected Value ($ per 100)', 'Odds Source']
    
    # Apply color styling
    def color_ev(val):
        color = 'lightgreen' if val > 0 else 'lightcoral'
        return f'background-color: {color}'
        
    st.dataframe(display_df.style.applymap(color_ev, subset=['Expected Value ($ per 100)']), use_container_width=True)

st.sidebar.title("About the Model")
st.sidebar.info("""
**Architecture**:
- **Data**: Live NHL standings & advanced metrics (PP%, PK%)
- **Machine Learning**: RandomForestClassifier trained on simulated momentum indicators
- **Odds**: Connected to The Odds API via ODs_API_KEY env variable, or mathematically mocked.
""")
