import streamlit as st
import pandas as pd
from predictor import ProfessionalNHLPredictor

st.set_page_config(page_title="NHL Betting Predictor", page_icon="🏒", layout="wide")

st.title("🏒 Pro NHL Betting Predictor")
st.caption("Advanced **XGBoost + Poisson Distribution** engine trained on 6,700 real games. Predicts Moneyline, Exact Score, and Over/Under +EV opportunities.")

@st.cache_resource()
def load_predictor():
    app = ProfessionalNHLPredictor()
    app.train_real_model()
    return app

predictor = load_predictor()

st.sidebar.title("💰 Bankroll Management")
st.sidebar.caption("The predictor uses the **Kelly Criterion** to mathematically calculate your optimal bet size to maximize compound growth while preventing bankruptcy.")
user_bankroll = st.sidebar.number_input("Your Total Bankroll ($)", min_value=10, max_value=1000000, value=1000, step=50)

st.header("Today's NHL Predictions")

if st.button("🔄 Refresh Live Odds & Predictions"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

@st.cache_data(ttl=3600)
def get_daily_predictions_v9():
    return predictor.run_daily_predictions()

with st.spinner("Fetching live NHL stats, training models, and pulling odds..."):
    results = get_daily_predictions_v9()

if not results:
    st.info("No NHL games are scheduled for today, or data could not be retrieved.")
else:
    # ---- High Value Summary Banner ----
    ev_bets = []
    for res in results:
        if res.get('ev', 0) > 0:
            rec_wager = user_bankroll * res.get('kelly_ml', 0)
            ev_bets.append(f"**{res['predicted_winner']}** ML (Bet ${rec_wager:.2f})")
        if res.get('ev_over', 0) > 0:
            rec_wager = user_bankroll * res.get('kelly_over', 0)
            ev_bets.append(f"**OVER {res['o_u_line']}** {res['matchup']} (Bet ${rec_wager:.2f})")
        if res.get('ev_under', 0) > 0:
            rec_wager = user_bankroll * res.get('kelly_under', 0)
            ev_bets.append(f"**UNDER {res['o_u_line']}** {res['matchup']} (Bet ${rec_wager:.2f})")

    if ev_bets:
        st.success("✅ **Today's +EV Bets:** " + "  |  ".join(ev_bets))
    else:
        st.warning("⚠️ No +EV bets found today. Proceed with caution.")
    
    st.divider()

    # ---- Per-Game Cards ----
    for res in results:
        with st.container():
            exact_sc = res.get('exact_score', 'N/A')
            src_badge = "🟢 Live API" if res.get('data_source') == 'Odds API' else "🟡 Mocked"
            st.markdown(f"### {res['matchup']}   |   🎯 Pred. Score: **{exact_sc}**   {src_badge}")
            
            # Formulate Phase 9 Injury Warnings
            injury_text = []
            away_abbrev, home_abbrev = res['matchup'].split(' @ ')
            if res.get('away_injury_penalty', 0) > 0.05:
                injury_text.append(f"{away_abbrev} (-{res['away_injury_penalty']*100:.0f}% xG)")
            if res.get('home_injury_penalty', 0) > 0.05:
                injury_text.append(f"{home_abbrev} (-{res['home_injury_penalty']*100:.0f}% xG)")
            inj_str = f"  |  🏥 **Injuries:** {', '.join(injury_text)}" if injury_text else ""
            
            # Formulate Phase 10 Special Teams Edge
            st_text = []
            if res.get('away_st_disparity', 0) > 0.1:
                st_text.append(f"{away_abbrev} (+{res['away_st_disparity']:.2f} PP Edge)")
            elif res.get('away_st_disparity', 0) < -0.1:
                st_text.append(f"{away_abbrev} ({res['away_st_disparity']:.2f} PP Edge)")
                
            if res.get('home_st_disparity', 0) > 0.1:
                st_text.append(f"{home_abbrev} (+{res['home_st_disparity']:.2f} PP Edge)")
            elif res.get('home_st_disparity', 0) < -0.1:
                st_text.append(f"{home_abbrev} ({res['home_st_disparity']:.2f} PP Edge)")
                
            st_str = f"  |  ⚡ **Special Teams:** {', '.join(st_text)}" if st_text else ""
            
            st.caption(f"📅 {res.get('date', '')}  |  🥅 **Goalies:** {res.get('away_goalie', 'Team Avg')} vs {res.get('home_goalie', 'Team Avg')}{inj_str}{st_str}")

            # Moneyline
            st.markdown("##### Moneyline")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Winner", res['predicted_winner'])
            with col2:
                st.metric("Model Confidence", res['confidence'])
            with col3:
                st.metric("Decimal Odds", res['odds'])

            ev_ml = res.get('ev', 0)
            if ev_ml > 0:
                rec_wager = user_bankroll * res.get('kelly_ml', 0)
                st.success(f"📈 **Moneyline Value Bet!** EV: +${ev_ml:.2f} per $100  |  💰 **Rec. Wager:** ${rec_wager:.2f} ({res.get('kelly_ml', 0)*100:.1f}%)")
            else:
                st.warning(f"📉 **No ML Edge.** EV: ${ev_ml:.2f} per $100. Skip.")

            # Over/Under
            st.markdown("##### Over/Under (Totals)")
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Vegas O/U Line", f"{res.get('o_u_line', 'N/A')}")
            with col5:
                st.metric("Model Projected Total", f"{res.get('projected_total', 'N/A')} goals")
            with col6:
                st.metric("Live O/U Odds", f"O {res.get('over_odds','?')}  /  U {res.get('under_odds','?')}")

            ev_over = res.get('ev_over', 0)
            ev_under = res.get('ev_under', 0)
            if ev_over > 0:
                rec_wager = user_bankroll * res.get('kelly_over', 0)
                st.success(f"📈 **OVER {res.get('o_u_line')} is a Value Bet!** EV: +${ev_over:.2f}  |  💰 **Rec. Wager:** ${rec_wager:.2f}")
            elif ev_under > 0:
                rec_wager = user_bankroll * res.get('kelly_under', 0)
                st.success(f"📈 **UNDER {res.get('o_u_line')} is a Value Bet!** EV: +${ev_under:.2f}  |  💰 **Rec. Wager:** ${rec_wager:.2f}")
            else:
                st.warning(f"📉 **No O/U Edge.** (Over EV: ${ev_over:.2f} | Under EV: ${ev_under:.2f}). Skip.")

            st.divider()

    # ---- Full Summary Table ----
    st.subheader("📊 Full Game Summary")
    table_rows = []
    for res in results:
        table_rows.append({
            "Matchup": res['matchup'],
            "Goalies": f"{res.get('away_goalie', 'Avg')} @ {res.get('home_goalie', 'Avg')}",
            "Pred. Winner": res['predicted_winner'],
            "Confidence": res['confidence'],
            "Exact Score": res.get('exact_score', 'N/A'),
            "ML EV ($)": round(res.get('ev', 0), 2),
            "O/U Line": res.get('o_u_line', 'N/A'),
            "Proj. Goals": res.get('projected_total', 'N/A'),
            "Over EV ($)": round(res.get('ev_over', 0), 2),
            "Under EV ($)": round(res.get('ev_under', 0), 2),
        })
    summary_df = pd.DataFrame(table_rows)

    def highlight_ev(val):
        try:
            color = 'background-color: #1a7a1a; color: white' if float(val) > 0 else ''
            return color
        except:
            return ''

    st.dataframe(
        summary_df.style.applymap(highlight_ev, subset=['ML EV ($)', 'Over EV ($)', 'Under EV ($)']),
        use_container_width=True
    )

st.sidebar.divider()
st.sidebar.title("About the Model")
st.sidebar.info("""
**Architecture**:
- **Data**: Live NHL API + MoneyPuck deep analytics (Corsi, Fenwick, High-Danger xG)
- **ML**: XGBoost – 62.4% cross-validated accuracy on 6,700 real games
- **Score Engine**: Poisson Distribution exact score simulator
- **Betting**: Moneyline, Over/Under, Kelly Criterion, and full +EV analysis
- **Odds**: The Odds API (live) or mathematical mock fallback
""")

