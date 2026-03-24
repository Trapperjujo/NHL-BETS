import pandas as pd
import numpy as np
import datetime
from xgboost import XGBClassifier

from data_fetcher import NHLDataFetcher
from feature_engine import FeatureEngine
from odds_integrator import OddsIntegrator

class ProfessionalNHLPredictor:
    def __init__(self):
        self.fetcher = NHLDataFetcher()
        self.engine = FeatureEngine()
        self.odds = OddsIntegrator()
        
        # Phase 4 Upgrade: XGBoost for sharper probabilistic outputs
        self.model = XGBClassifier(
            n_estimators=300, 
            max_depth=5, 
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
        self.is_trained = False

    def train_real_model(self):
        """
        Loads the massive historical dataset compiled by historical_data_builder.py
        and trains the predictive model on thousands of real game outcomes.
        """
        print("Loading real historical NHL game database...")
        try:
            df = pd.read_csv("historical_training_data.csv")
            print(f"Loaded {len(df)} historical matchups. Training algorithm...")
            
            # The database contains most features, but we need to supply the 
            # advanced Phase 2 metrics (PP%, PK%, Streak) that MoneyPuck's CSV 
            # lacked, so the model knows how to weigh them when it sees them in real-time.
            # We approximate historical PP%/PK% variance by correlating it directly to team goal scoring.
            
            df['home_streak'] = 0 # Baseline
            df['away_streak'] = 0
            df['home_pp_pct'] = 20.0 + (df['home_gf_pg'] - 3.0) * 5.0
            df['away_pp_pct'] = 20.0 + (df['away_gf_pg'] - 3.0) * 5.0
            df['home_pk_pct'] = 80.0 - (df['home_ga_pg'] - 3.0) * 5.0
            df['away_pk_pct'] = 80.0 - (df['away_ga_pg'] - 3.0) * 5.0
            
            X = df[[
                'home_win_pct', 'away_win_pct', 'home_gf_pg', 'away_gf_pg',
                'home_ga_pg', 'away_ga_pg', 'home_l10_win_pct', 'away_l10_win_pct',
                'home_streak', 'away_streak', 'home_pp_pct', 'away_pp_pct',
                'home_pk_pct', 'away_pk_pct', 'home_shots_diff', 'away_shots_diff',
                'home_xg_for_pg', 'away_xg_for_pg', 'home_xg_against_pg', 'away_xg_against_pg',
                'home_sv_pct', 'away_sv_pct', 'home_is_b2b', 'away_is_b2b'
            ]]
            y = df['home_win']
            
            self.model.fit(X, y)
            self.is_trained = True
            print("Model successfully trained on historical data!\n")
            
        except FileNotFoundError:
            print("ERROR: historical_training_data.csv not found! Run historical_data_builder.py first.")


    def run_daily_predictions(self):
        if not self.is_trained:
            self.train_real_model()

        print("Fetching real-time NHL Standings and Advanced Stats...")
        standings = self.fetcher.fetch_current_standings()
        advanced_stats = self.fetcher.fetch_advanced_stats()
        schedule = self.fetcher.fetch_todays_schedule()
        
        print("Fetching real-time Extreme Deep Metrics (Fatigue, xG, SV%)...")
        tired_teams = self.fetcher.fetch_tired_teams()
        mp_stats = self.fetcher.fetch_moneypuck_stats()
        
        if not schedule:
            print("No games scheduled today or data unavailable.")
            return []

        team_features = self.engine.build_team_features(
            standings_data=standings, 
            advanced_stats_data=advanced_stats,
            moneypuck_stats=mp_stats,
            tired_teams=tired_teams
        )
        print(f"Engineered features for {len(team_features)} teams.\n")

        print("Fetching today's schedule...")
        games = schedule # Use the schedule fetched earlier
        
        if not games:
            print("No games scheduled.")
            return []

        print(f"Found {len(games)} game(s). Generating predictions and +EV analysis...\n")
        print("="*60)
        
        results = []

        for game in games:
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            start_time_utc = game.get('startTimeUTC', 'Unknown')
            
            # Format the time (Basic string parsing since we know ISO format)
            try:
                dt = datetime.datetime.strptime(start_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                formatted_time = dt.strftime("%A, %b %d at %H:%M UTC")
            except:
                formatted_time = start_time_utc
            
            home_abbrev = home_team.get('abbrev')
            away_abbrev = away_team.get('abbrev')
            
            if not home_abbrev or not away_abbrev:
                continue
                
            matchup_features = self.engine.get_matchup_features(home_abbrev, away_abbrev, team_features)
            
            if matchup_features is None:
                print(f"Skipping {away_abbrev} @ {home_abbrev}: Missing team stats.")
                continue

            # 1. Predict Outcome
            prob = self.model.predict_proba(matchup_features)[0]
            home_prob = prob[1]
            away_prob = prob[0]
            
            # 2. Mock Live Odds Integration
            home_full_name = team_features.loc[home_abbrev]['team_name'] if home_abbrev in team_features.index else home_abbrev
            away_full_name = team_features.loc[away_abbrev]['team_name'] if away_abbrev in team_features.index else away_abbrev
            odds_data = self.odds.fetch_live_odds(home_full_name, away_full_name, home_prob)
            home_odds = odds_data['home_odds']
            away_odds = odds_data['away_odds']
            data_source = "(Real API)" if odds_data.get('is_real_data') else "(Mocked)"
            
            # 3. Choose bet based on highest prob, calculate +EV
            if home_prob > away_prob:
                predicted_winner = home_abbrev
                model_confidence = home_prob
                suggested_odds = home_odds
                ev = self.odds.calculate_ev(home_prob, home_odds)
            else:
                predicted_winner = away_abbrev
                model_confidence = away_prob
                suggested_odds = away_odds
                ev = self.odds.calculate_ev(away_prob, away_odds)

            # 4. Display results
            print(f"MATCHUP: {away_abbrev} @ {home_abbrev} [{formatted_time}]")
            print(f"  Predicted Winner : {predicted_winner} ({model_confidence*100:.1f}%)")
            print(f"  Live Odds        : {predicted_winner} @ {suggested_odds} {data_source}")
            
            if ev > 0:
                print(f"  Analysis         : POSITIVE EV (+${ev:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  Analysis         : NEGATIVE EV (${ev:.2f} per $100 bet) [SKIP] Skip")
            print("-" * 60)
            
            results.append({
                'matchup': f"{away_abbrev} @ {home_abbrev}",
                'date': formatted_time,
                'predicted_winner': predicted_winner,
                'confidence': f"{model_confidence*100:.1f}%",
                'odds': suggested_odds,
                'ev': ev,
                'is_value': ev > 0,
                'data_source': data_source
            })
            
        return results

if __name__ == "__main__":
    print("Welcome to the Pro NHL Predictor System")
    print("-" * 40)
    app = ProfessionalNHLPredictor()
    app.run_daily_predictions()
