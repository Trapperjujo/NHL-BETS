import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data_fetcher import NHLDataFetcher
from feature_engine import FeatureEngine
from odds_integrator import OddsIntegrator

class ProfessionalNHLPredictor:
    def __init__(self):
        self.fetcher = NHLDataFetcher()
        self.engine = FeatureEngine()
        self.odds = OddsIntegrator()
        
        # We use a Random Forest which can capture non-linear interactions
        self.model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        self.is_trained = False

    def train_synthetic_model(self):
        """
        Since we don't have a massive historical database of point-in-time stats 
        readily available from the free API, we generate a synthetic dataset
        that represents typical hockey dynamics to train our model.
        
        In production, you would replace this with a real historical dataset (e.g., CSV).
        """
        print("Training model on synthetic historical data...")
        np.random.seed(42)
        n_samples = 2000
        
        # Generate realistic-looking team stats for historical matchups
        home_win_pct = np.random.uniform(0.35, 0.75, n_samples)
        away_win_pct = np.random.uniform(0.35, 0.75, n_samples)
        home_gf_pg = np.random.uniform(2.2, 3.8, n_samples)
        away_gf_pg = np.random.uniform(2.2, 3.8, n_samples)
        home_ga_pg = np.random.uniform(2.2, 3.8, n_samples)
        away_ga_pg = np.random.uniform(2.2, 3.8, n_samples)
        home_l10 = np.random.uniform(0.2, 0.8, n_samples)
        away_l10 = np.random.uniform(0.2, 0.8, n_samples)
        home_streak = np.random.randint(-5, 6, n_samples)
        away_streak = np.random.randint(-5, 6, n_samples)
        
        # Advanced Stats (Simulated)
        home_pp_pct = np.random.uniform(15.0, 28.0, n_samples)  # 15% to 28%
        away_pp_pct = np.random.uniform(15.0, 28.0, n_samples)
        home_pk_pct = np.random.uniform(75.0, 88.0, n_samples)  # 75% to 88%
        away_pk_pct = np.random.uniform(75.0, 88.0, n_samples)
        home_shots_diff = np.random.uniform(-5.0, 5.0, n_samples)
        away_shots_diff = np.random.uniform(-5.0, 5.0, n_samples)
        
        # Calculate a 'true' probability based on these stats
        # Home ice advantage adds a small bump (~4-5%)
        base_prob = 0.54 
        
        # Influence logic: better win pct, better goal diff, better recent form, better special teams
        prob_adjustment = (
            (home_win_pct - away_win_pct) * 0.7 + 
            ((home_gf_pg - home_ga_pg) - (away_gf_pg - away_ga_pg)) * 0.1 +
            (home_l10 - away_l10) * 0.3 +
            (home_streak - away_streak) * 0.02 +
            ((home_pp_pct + home_pk_pct) - (away_pp_pct + away_pk_pct)) * 0.005 +
            (home_shots_diff - away_shots_diff) * 0.01
        )
        
        final_probs = np.clip(base_prob + prob_adjustment, 0.1, 0.9)
        
        # Determine actual winner based on the probability
        outcomes = np.random.binomial(1, final_probs)
        
        X = pd.DataFrame({
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct,
            'home_gf_pg': home_gf_pg,
            'away_gf_pg': away_gf_pg,
            'home_ga_pg': home_ga_pg,
            'away_ga_pg': away_ga_pg,
            'home_l10_win_pct': home_l10,
            'away_l10_win_pct': away_l10,
            'home_streak': home_streak,
            'away_streak': away_streak,
            'home_pp_pct': home_pp_pct,
            'away_pp_pct': away_pp_pct,
            'home_pk_pct': home_pk_pct,
            'away_pk_pct': away_pk_pct,
            'home_shots_diff': home_shots_diff,
            'away_shots_diff': away_shots_diff
        })
        y = outcomes
        
        self.model.fit(X, y)
        self.is_trained = True
        print("Model training complete.\n")

    def run_daily_predictions(self):
        if not self.is_trained:
            self.train_synthetic_model()

        print("Fetching real-time NHL Standings and Advanced Stats...")
        standings = self.fetcher.fetch_current_standings()
        advanced_stats = self.fetcher.fetch_advanced_stats()
        
        if not standings:
            print("Failed to fetch standings.")
            return

        team_features = self.engine.build_team_features(standings, advanced_stats)
        print(f"Engineered features for {len(team_features)} teams.\n")

        print("Fetching today's schedule...")
        games = self.fetcher.fetch_todays_schedule()
        
        if not games:
            print("No games scheduled.")
            return []

        print(f"Found {len(games)} game(s). Generating predictions and +EV analysis...\n")
        print("="*60)
        
        results = []

        for game in games:
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
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
            print(f"MATCHUP: {away_abbrev} @ {home_abbrev}")
            print(f"  Predicted Winner : {predicted_winner} ({model_confidence*100:.1f}%)")
            print(f"  Live Odds        : {predicted_winner} @ {suggested_odds} {data_source}")
            
            if ev > 0:
                print(f"  Analysis         : POSITIVE EV (+${ev:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  Analysis         : NEGATIVE EV (${ev:.2f} per $100 bet) [SKIP] Skip")
            print("-" * 60)
            
            results.append({
                'matchup': f"{away_abbrev} @ {home_abbrev}",
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
