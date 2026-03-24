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
        
        # Phase 6 mathematically optimal XGBoost parameters (from exhaustive 36-feature Grid Search)
        self.model = XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=5,
            n_estimators=200,
            subsample=1.0,
            eval_metric="logloss",
            random_state=42
        )
        self.is_trained = False

    def train_real_model(self):
        """
        Loads the massive historical dataset compiled by historical_data_builder.py
        and trains the predictive model on thousands of real game outcomes.
        """
        import joblib, os
        model_path = "nhl_model.pkl"
        
        # Fast path: load a previously saved model from disk
        if os.path.exists(model_path):
            print("Loading cached model from disk (nhl_model.pkl)...")
            self.model = joblib.load(model_path)
            self.is_trained = True
            print("Model loaded instantly from cache!")
            return
        
        print("Loading real historical NHL game database...")
        try:
            df = pd.read_csv("historical_training_data.csv")
            print(f"Loaded {len(df)} historical matchups. Training algorithm...")
            
            df['home_streak'] = 0
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
                'home_sv_pct', 'away_sv_pct', 'home_is_b2b', 'away_is_b2b',
                'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
                'home_hd_shots_for', 'away_hd_shots_for', 'home_hd_shots_against', 'away_hd_shots_against',
                'home_hd_xg_for', 'away_hd_xg_for', 'home_hd_xg_against', 'away_hd_xg_against',
                'home_sva_xg_for', 'away_sva_xg_for', 'home_sva_xg_against', 'away_sva_xg_against',
                'home_pen_drawn', 'away_pen_drawn', 'home_pen_taken', 'away_pen_taken'
            ]]
            y = df['home_win']
            
            self.model.fit(X, y)
            self.is_trained = True
            print("Model successfully trained on historical data!\n")
            
            # Save for instant load next time
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path} for fast future loads.")
            
        except FileNotFoundError:
            print("ERROR: historical_training_data.csv not found! Run historical_data_builder.py first.")

    def calculate_poisson_ou(self, expected_total_goals, ou_line):
        """
        Uses a Poisson distribution to calculate the probability of the game going
        UNDER or OVER the sportsbook line based on our Model's Expected Total Goals.
        """
        import math
        prob_under = 0.0
        max_under_goals = math.floor(ou_line)
        
        for k in range(max_under_goals + 1):
            prob = (math.exp(-expected_total_goals) * (expected_total_goals ** k)) / math.factorial(k)
            prob_under += prob
            
        prob_over = 1.0 - prob_under
        return prob_over, prob_under

    def predict_exact_score(self, home_xg, away_xg):
        """
        Calculates the most mathematically probable exact final score.
        Iterates through 100 possible score combinations using independent Poisson distributions.
        Excludes ties since NHL games always end with a winner.
        """
        import math
        best_score = (0, 0)
        max_prob = 0.0
        
        for h in range(10):
            for a in range(10):
                if h == a:
                    continue # Games don't end in ties
                prob_h = (math.exp(-home_xg) * (home_xg ** h)) / math.factorial(h)
                prob_a = (math.exp(-away_xg) * (away_xg ** a)) / math.factorial(a)
                prob_score = prob_h * prob_a
                
                if prob_score > max_prob:
                    max_prob = prob_score
                    best_score = (h, a)
                    
        return best_score

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
            home_team_data = game.get('homeTeam', {})
            away_team_data = game.get('awayTeam', {})
            start_time_utc = game.get('startTimeUTC', 'Unknown')
            
            # Format the time (Basic string parsing since we know ISO format)
            try:
                dt = datetime.datetime.strptime(start_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                formatted_time = dt.strftime("%A, %b %d at %H:%M UTC")
            except:
                formatted_time = start_time_utc
            
            home_abbrev = home_team_data.get('abbrev')
            away_abbrev = away_team_data.get('abbrev')
            
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
            
            # Fetch odds for moneyline and O/U
            odds_data = self.odds.fetch_live_odds(home_full_name, away_full_name, home_prob)
            
            # 3. Choose bet based on highest prob, calculate +EV
            # Moneyline EV
            ev_home = self.odds.calculate_ev(home_prob, odds_data['home_odds'])
            ev_away = self.odds.calculate_ev(away_prob, odds_data['away_odds'])
            
            # Poisson Over/Under Calculations
            # Phase 7 Upgrade: Opponent-adjusted goal projection
            # Instead of raw xGF, we blend each team's offense with the opponent's defense
            # and factor in NHL home ice advantage (~7.5% more goals for home teams)
            HOME_ADVANTAGE = 1.075
            
            home_xgf = matchup_features['home_xg_for_pg'].values[0]
            away_xga = matchup_features['away_xg_against_pg'].values[0]
            away_xgf = matchup_features['away_xg_for_pg'].values[0]
            home_xga = matchup_features['home_xg_against_pg'].values[0]
            
            home_proj_goals = ((home_xgf + away_xga) / 2) * HOME_ADVANTAGE
            away_proj_goals = (away_xgf + home_xga) / 2
            model_projected_total = home_proj_goals + away_proj_goals
            
            prob_over, prob_under = self.calculate_poisson_ou(model_projected_total, odds_data['o_u_line'])
            
            ev_over = self.odds.calculate_ev(prob_over, odds_data['over_odds']) if odds_data.get('over_odds') else 0
            ev_under = self.odds.calculate_ev(prob_under, odds_data['under_odds']) if odds_data.get('under_odds') else 0
            
            # Exact Score Prediction
            exact_home, exact_away = self.predict_exact_score(home_proj_goals, away_proj_goals)

            # 4. Display results
            print(f"MATCHUP: {away_abbrev} @ {home_abbrev} [{formatted_time}]")
            
            # Moneyline Analysis
            predicted_winner_abbrev = home_abbrev if home_prob > away_prob else away_abbrev
            model_confidence = max(home_prob, away_prob)
            suggested_odds_ml = odds_data['home_odds'] if home_prob > away_prob else odds_data['away_odds']
            ev_ml = ev_home if home_prob > away_prob else ev_away
            
            print(f"  Predicted Winner : {predicted_winner_abbrev} ({model_confidence*100:.1f}%)")
            print(f"  Exact Score Pred : {exact_away} - {exact_home}")
            print(f"  Live ML Odds     : {predicted_winner_abbrev} @ {suggested_odds_ml} {'(Real API)' if odds_data.get('is_real_data') else '(Mocked)'}")
            if ev_ml > 0:
                print(f"  ML Analysis      : POSITIVE EV (+${ev_ml:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  ML Analysis      : NEGATIVE EV (${ev_ml:.2f} per $100 bet) [SKIP] Skip")

            # O/U Analysis
            print(f"  Projected Total  : {model_projected_total:.2f} (O/U Line: {odds_data['o_u_line']})")
            print(f"  Live O/U Odds    : Over {odds_data['o_u_line']} @ {odds_data['over_odds']} / Under {odds_data['o_u_line']} @ {odds_data['under_odds']}")
            if ev_over > 0:
                print(f"  O/U Analysis (O) : POSITIVE EV (+${ev_over:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  O/U Analysis (O) : NEGATIVE EV (${ev_over:.2f} per $100 bet) [SKIP] Skip")
            if ev_under > 0:
                print(f"  O/U Analysis (U) : POSITIVE EV (+${ev_under:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  O/U Analysis (U) : NEGATIVE EV (${ev_under:.2f} per $100 bet) [SKIP] Skip")
            print("-" * 60)
            
            results.append({
                'matchup': f"{away_abbrev} @ {home_abbrev}",
                'date': formatted_time,
                # Moneyline
                'predicted_winner': predicted_winner_abbrev,
                'confidence': f"{model_confidence*100:.1f}%",
                'exact_score': f"{exact_away} - {exact_home}",
                'odds': suggested_odds_ml,
                'ev': ev_ml,
                
                # Totals (O/U)
                'o_u_line': odds_data.get('o_u_line', 6.5),
                'projected_total': round(float(model_projected_total), 2),
                'over_odds': odds_data.get('over_odds', 1.90),
                'under_odds': odds_data.get('under_odds', 1.90),
                'ev_over': ev_over,
                'ev_under': ev_under,
                
                'data_source': "Odds API" if odds_data.get('is_real_data') else "Mocked"
            })
            
        return results

if __name__ == "__main__":
    print("Welcome to the Pro NHL Predictor System")
    print("-" * 40)
    app = ProfessionalNHLPredictor()
    app.run_daily_predictions()
