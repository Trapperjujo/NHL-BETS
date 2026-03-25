import pandas as pd
import numpy as np
import datetime
from xgboost import XGBClassifier

from data_fetcher import NHLDataFetcher
from feature_engine import FeatureEngine
from odds_integrator import OddsIntegrator
from line_movement_tracker import LineMovementTracker

class ProfessionalNHLPredictor:
    def __init__(self):
        self.fetcher = NHLDataFetcher()
        self.engine = FeatureEngine()
        self.odds = OddsIntegrator()
        self.steam = LineMovementTracker()
        
        # Phase 5 Ultimate Syndicate optimal XGBoost parameters (Tuned via 50-cycle Optuna run)
        self.model = XGBClassifier(
            colsample_bytree=0.7238,
            learning_rate=0.00141,
            max_depth=3,
            min_child_weight=5,
            n_estimators=1000,
            subsample=0.8183,
            gamma=0.0129,
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
        model_path = "nhl_model_v4_phase5.pkl"
        
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
            
            # Phase 5: HDSV% and Flight Fatigue Backfill
            df['home_hdsv_pct'] = df['home_sv_pct'] - 0.080
            df['away_hdsv_pct'] = df['away_sv_pct'] - 0.080
            df['home_flight_fatigue'] = 1.0
            df['away_flight_fatigue'] = 1.0
            
            X = df[[
                'home_win_pct', 'away_win_pct', 'home_gf_pg', 'away_gf_pg',
                'home_ga_pg', 'away_ga_pg', 'home_pp_pct', 'away_pp_pct',
                'home_pk_pct', 'away_pk_pct', 'home_shots_diff', 'away_shots_diff',
                'home_xg_for_pg', 'away_xg_for_pg', 'home_xg_against_pg', 'away_xg_against_pg',
                'home_sv_pct', 'away_sv_pct', 'home_hdsv_pct', 'away_hdsv_pct',
                'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
                'home_hd_shots_for', 'away_hd_shots_for', 'home_hd_shots_against', 'away_hd_shots_against',
                'home_hd_xg_for', 'away_hd_xg_for', 'home_hd_xg_against', 'away_hd_xg_against',
                'home_sva_xg_for', 'away_sva_xg_for', 'home_sva_xg_against', 'away_sva_xg_against',
                'home_pen_drawn', 'away_pen_drawn', 'home_pen_taken', 'away_pen_taken',
                'home_elo', 'away_elo', 'home_flight_fatigue', 'away_flight_fatigue'
            ]]
            y = df['home_win']
            
            from sklearn.calibration import CalibratedClassifierCV
            # Wrap the XGBoost model in Platt Sigmoid Scaling for smoother, wider confidence spreads
            self.model = CalibratedClassifierCV(estimator=self.model, method='sigmoid', cv=5)
            
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

    def calculate_poisson_spread(self, home_xg, away_xg, home_spread_line):
        """
        Calculates the exact mathematical probability of covering a Puck Line Spread.
        Iterates through dual Poisson distributions to find the margin of victory.
        """
        import math
        prob_home_cover = 0.0
        prob_total_valid = 0.0
        
        for h in range(15):
            for a in range(15):
                if h == a: continue # NHL games don't tie
                
                prob_h = (math.exp(-home_xg) * (home_xg ** h)) / math.factorial(h)
                prob_a = (math.exp(-away_xg) * (away_xg ** a)) / math.factorial(a)
                prob_score = prob_h * prob_a
                
                prob_total_valid += prob_score
                
                if (h - a) > home_spread_line:
                    prob_home_cover += prob_score
                    
        # Normalize to exclude ties (since we don't count h==a)
        if prob_total_valid > 0:
            prob_home_cover = prob_home_cover / prob_total_valid
            
        prob_away_cover = 1.0 - prob_home_cover
        return prob_home_cover, prob_away_cover

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
        mp_stats = self.fetcher.fetch_moneypuck_stats()
        
        print("Scraping Confirmed Starting Goalies (Phase 9)...")
        starting_goalies = self.fetcher.fetch_starting_goalies()
        
        print("Scraping Active Roster Injuries & Scratches (Phase 9)...")
        all_teams = []
        if schedule:
            for g in schedule:
                all_teams.append(g.get('homeTeam', {}).get('abbrev'))
                all_teams.append(g.get('awayTeam', {}).get('abbrev'))
                
        # Scrape Injuries
        injury_impacts = self.fetcher.fetch_injury_impacts(list(set(filter(None, all_teams))))
        
        # Phase 10: Travel & Rest Calculus
        print("Calculating exact Travel & Rest Disparity (Phase 10)...")
        rest_days_dict = self.fetcher.fetch_rest_days(list(set(filter(None, all_teams))))

        if not schedule:
            print("No games scheduled today or data unavailable.")
            return []

        team_features = self.engine.build_team_features(
            standings_data=standings, 
            advanced_stats_data=advanced_stats,
            moneypuck_stats=mp_stats,
            rest_days_dict=rest_days_dict,
            starting_goalies=starting_goalies,
            injury_impacts=injury_impacts
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
            
            # Format the time to Winnipeg Central Time (America/Winnipeg)
            try:
                import pytz
                dt_utc = datetime.datetime.strptime(start_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
                winnipeg_tz = pytz.timezone("America/Winnipeg")
                dt_winnipeg = dt_utc.astimezone(winnipeg_tz)
                formatted_time = dt_winnipeg.strftime("%A, %b %d at %I:%M %p CT")
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

            # Extract metadata (strings/non-training floats) before pushing to XGBoost matrix
            home_goalie = matchup_features['home_goalie'].values[0] if 'home_goalie' in matchup_features.columns else 'Team Avg'
            away_goalie = matchup_features['away_goalie'].values[0] if 'away_goalie' in matchup_features.columns else 'Team Avg'
            home_injury_penalty = matchup_features['home_injury_penalty'].values[0] if 'home_injury_penalty' in matchup_features.columns else 0.0
            away_injury_penalty = matchup_features['away_injury_penalty'].values[0] if 'away_injury_penalty' in matchup_features.columns else 0.0
            
            home_rest = matchup_features['home_rest_days'].values[0] if 'home_rest_days' in matchup_features.columns else 2
            away_rest = matchup_features['away_rest_days'].values[0] if 'away_rest_days' in matchup_features.columns else 2
            
            # CRITICAL FIX: Ensure X_live exactly matches the model's expected features
            # Define the exact 44 features used in training order
            EXPECTED_FEATURES = [
                'home_win_pct', 'away_win_pct', 'home_gf_pg', 'away_gf_pg',
                'home_ga_pg', 'away_ga_pg', 'home_pp_pct', 'away_pp_pct',
                'home_pk_pct', 'away_pk_pct', 'home_shots_diff', 'away_shots_diff',
                'home_xg_for_pg', 'away_xg_for_pg', 'home_xg_against_pg', 'away_xg_against_pg',
                'home_sv_pct', 'away_sv_pct', 'home_hdsv_pct', 'away_hdsv_pct',
                'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
                'home_hd_shots_for', 'away_hd_shots_for', 'home_hd_shots_against', 'away_hd_shots_against',
                'home_hd_xg_for', 'away_hd_xg_for', 'home_hd_xg_against', 'away_hd_xg_against',
                'home_sva_xg_for', 'away_sva_xg_for', 'home_sva_xg_against', 'away_sva_xg_against',
                'home_pen_drawn', 'away_pen_drawn', 'home_pen_taken', 'away_pen_taken',
                'home_elo', 'away_elo', 'home_flight_fatigue', 'away_flight_fatigue'
            ]
            
            # If the loaded model has feature names, align to them (handles stale models gracefully)
            model_features = None
            if hasattr(self.model, 'estimator') and hasattr(self.model.estimator, 'feature_names_in_'):
                model_features = self.model.estimator.feature_names_in_
            elif hasattr(self.model, 'feature_names_in_'):
                model_features = self.model.feature_names_in_
                
            if model_features is not None:
                # Reindex with fill_value=0 to pad missing columns if using a stale model
                X_live = matchup_features.reindex(columns=model_features, fill_value=0)
            else:
                # Fallback to the hardcoded list
                X_live = matchup_features.reindex(columns=EXPECTED_FEATURES, fill_value=0)

            # 1. Predict Outcome
            prob = self.model.predict_proba(X_live)[0]
            home_prob = prob[1]
            away_prob = prob[0]
            
            # 2. Mock Live Odds Integration
            home_full_name = team_features.loc[home_abbrev]['team_name'] if home_abbrev in team_features.index else home_abbrev
            away_full_name = team_features.loc[away_abbrev]['team_name'] if away_abbrev in team_features.index else away_abbrev
            
            # Fetch odds for moneyline and O/U
            odds_data = self.odds.fetch_live_odds(home_full_name, away_full_name, home_prob)
            
            # Enforce strict real-time API odds; skip generating mocked odds for games
            if not odds_data or not odds_data.get('is_real_data'):
                print(f"Skipping {away_abbrev} @ {home_abbrev} - No active odds on The Odds API.")
                continue
            
            # Phase 5: Steam Tracking (Reverse Line Movement)
            game_id = f"{away_abbrev}_{home_abbrev}"
            # If Pinnacle is present, use it for Steam tracking, otherwise fallback to standard best odds
            home_live_payout = odds_data.get('pin_true_home') or (1.0 / odds_data['home_odds'] if odds_data['home_odds'] > 0 else 0)
            away_live_payout = odds_data.get('pin_true_away') or (1.0 / odds_data['away_odds'] if odds_data['away_odds'] > 0 else 0)
            
            # We must convert probability back to decimal odds for the tracker, OR tracker accepts decimal odds natively
            # The tracker expects decimal odds format (e.g. 2.05). If we have true probability (like 0.55), odds = 1.0 / prob
            home_steam_odds = 1.0 / home_live_payout if home_live_payout > 0 else 0
            away_steam_odds = 1.0 / away_live_payout if away_live_payout > 0 else 0
            
            home_steam_data = self.steam.detect_steam(game_id, home_abbrev, home_steam_odds, public_bet_pct=0.45) # Mock public bet % for now
            away_steam_data = self.steam.detect_steam(game_id, away_abbrev, away_steam_odds, public_bet_pct=0.45)
            
            odds_data['home_steam'] = home_steam_data
            odds_data['away_steam'] = away_steam_data
            
            # 3. Choose bet based on highest prob, calculate +EV
            pin_true_home = odds_data.get('pin_true_home')
            pin_true_away = odds_data.get('pin_true_away')
            
            # Moneyline EV (Top-Down if Pinnacle is present, Bottom-Up otherwise)
            if pin_true_home is not None and pin_true_away is not None:
                ev_home = self.odds.calculate_ev(pin_true_home, odds_data['home_odds'])
                ev_away = self.odds.calculate_ev(pin_true_away, odds_data['away_odds'])
                home_prob = pin_true_home
                away_prob = pin_true_away
            else:
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
            
            # Phase 5: High-Danger Goalie Math
            # Calculate explicit matchup of High-Danger xG output vs the specific Goalie's HDSV%
            home_hd_xg = matchup_features['home_hd_xg_for'].values[0] if 'home_hd_xg_for' in matchup_features.columns else 1.0
            away_hd_xg = matchup_features['away_hd_xg_for'].values[0] if 'away_hd_xg_for' in matchup_features.columns else 1.0
            home_goalie_hdsv = matchup_features['home_hdsv_pct'].values[0] if 'home_hdsv_pct' in matchup_features.columns else 0.820
            away_goalie_hdsv = matchup_features['away_hdsv_pct'].values[0] if 'away_hdsv_pct' in matchup_features.columns else 0.820
            
            # If a goalie has 0.860 HDSV% (elite), they stop 4% more HD chances than the 0.820 average.
            # We scale the opponent's HD xG output by this exact differential!
            home_hd_advantage = (0.820 - away_goalie_hdsv) * home_hd_xg * 5.0 # Multiplier exposes the massive gravity of HD goals
            away_hd_advantage = (0.820 - home_goalie_hdsv) * away_hd_xg * 5.0

            home_proj_goals_base = (((home_xgf + home_hd_advantage) + away_xga) / 2) * HOME_ADVANTAGE
            away_proj_goals_base = ((away_xgf + away_hd_advantage) + home_xga) / 2
            
            # Phase 10: Special Teams Disparity Engine
            # Calculate the explicit mathematical mismatch between PP and PK
            home_pp = matchup_features['home_pp_pct'].values[0] / 100.0  # e.g., 0.25 (25%)
            away_pk = matchup_features['away_pk_pct'].values[0] / 100.0  # e.g., 0.80 (80%)
            away_pp = matchup_features['away_pp_pct'].values[0] / 100.0
            home_pk = matchup_features['home_pk_pct'].values[0] / 100.0
            
            home_pen_drawn = matchup_features['home_pen_drawn'].values[0]
            away_pen_drawn = matchup_features['away_pen_drawn'].values[0]
            
            # Expected powerplay goals = (Pens Drawn) * (PP Success Rate against this specific PK)
            # Baseline PK is ~80%. So if away_pk is 75%, home_pp gets a mathematical boost.
            home_pp_adj = home_pp * (0.80 / max(0.01, away_pk))
            away_pp_adj = away_pp * (0.80 / max(0.01, home_pk))
            
            # The raw goals generated specifically from this matchup's special teams
            home_pp_goals = home_pen_drawn * home_pp_adj
            away_pp_goals = away_pen_drawn * away_pp_adj
            
            # Average PP goals per game is already baked into xG (~0.5), so we extract the *marginal* edge
            home_st_disparity = (home_pp_goals - 0.5) * 0.40 # Dampen the multiplier to prevent exponential bleeding
            away_st_disparity = (away_pp_goals - 0.5) * 0.40
            
            # Phase 10: Travel & Rest Disparity Calculus
            # Compare the exact days of rest between the opponents
            rest_advantage = home_rest - away_rest
            
            # A completely rested team (3+ days) vs a back-to-back team (-2 days) generates roughly +0.2 marginal goal edge
            # Scale the penalty naturally: rest_advantage of +3 results in roughly +0.15 goals.
            home_rest_boost = rest_advantage * 0.05
            
            home_proj_goals = home_proj_goals_base + home_st_disparity + home_rest_boost
            away_proj_goals = away_proj_goals_base + away_st_disparity - home_rest_boost
            
            # Ensure goal projections don't break Poisson math (must be positive)
            home_proj_goals = max(0.1, home_proj_goals)
            away_proj_goals = max(0.1, away_proj_goals)
            
            model_projected_total = home_proj_goals + away_proj_goals
            
            prob_over, prob_under = self.calculate_poisson_ou(model_projected_total, odds_data['o_u_line'])
            
            ev_over = self.odds.calculate_ev(prob_over, odds_data['over_odds']) if odds_data.get('over_odds') else 0
            ev_under = self.odds.calculate_ev(prob_under, odds_data['under_odds']) if odds_data.get('under_odds') else 0
            
            # Phase 6 Spreads Calculations
            prob_home_cover, prob_away_cover = self.calculate_poisson_spread(home_proj_goals, away_proj_goals, odds_data['home_spread_line'])
            
            ev_home_spread = self.odds.calculate_ev(prob_home_cover, odds_data['home_spread_odds']) if odds_data.get('home_spread_odds') else 0
            ev_away_spread = self.odds.calculate_ev(prob_away_cover, odds_data['away_spread_odds']) if odds_data.get('away_spread_odds') else 0
            
            # Exact Score Prediction
            exact_home, exact_away = self.predict_exact_score(home_proj_goals, away_proj_goals)

            # Moneyline Analysis parameters
            predicted_winner_abbrev = home_abbrev if home_prob > away_prob else away_abbrev
            model_confidence = max(home_prob, away_prob)
            suggested_odds_ml = odds_data['home_odds'] if home_prob > away_prob else odds_data['away_odds']
            ev_ml = ev_home if home_prob > away_prob else ev_away

            # Phase 10: Kelly Criterion Fractional Bet Sizing
            # (Passing 1 as dummy bankroll since it now returns a pure percentage)
            kelly_ml = self.odds.calculate_kelly_criterion(model_confidence, suggested_odds_ml, 1.0) if ev_ml > 0 else 0.0
            kelly_over = self.odds.calculate_kelly_criterion(prob_over, odds_data['over_odds'], 1.0) if ev_over > 0 else 0.0
            kelly_under = self.odds.calculate_kelly_criterion(prob_under, odds_data['under_odds'], 1.0) if ev_under > 0 else 0.0
            
            kelly_home_spread = self.odds.calculate_kelly_criterion(prob_home_cover, odds_data['home_spread_odds'], 1.0) if ev_home_spread > 0 else 0.0
            kelly_away_spread = self.odds.calculate_kelly_criterion(prob_away_cover, odds_data['away_spread_odds'], 1.0) if ev_away_spread > 0 else 0.0

            # 4. Display results
            print(f"MATCHUP: {away_abbrev} @ {home_abbrev} [{formatted_time}]")
            
            print(f"  Goalie Matchup   : {away_goalie} vs {home_goalie}")
            print(f"  Predicted Winner : {predicted_winner_abbrev} ({model_confidence*100:.1f}%)")
            print(f"  Exact Score Pred : {exact_away} - {exact_home}")
            print(f"  Live ML Odds     : {predicted_winner_abbrev} @ {suggested_odds_ml} {'(Real API)' if odds_data.get('is_real_data') else '(Mocked)'}")
            if ev_ml > 0:
                print(f"  ML Analysis      : POSITIVE EV (+${ev_ml:.2f} per $100 bet) [YES] Value Bet!")
            else:
                print(f"  ML Analysis      : NEGATIVE EV (${ev_ml:.2f} per $100 bet) [SKIP] Skip")

            # Steam Analysis
            if odds_data.get('home_steam', {}).get('is_steam'):
                print(f"  [STEAM ALERT]    : Sharp Reverse Line Movement detected on {home_abbrev} (+{odds_data['home_steam']['delta_prob']*100:.1f}%)")
            if odds_data.get('away_steam', {}).get('is_steam'):
                print(f"  [STEAM ALERT]    : Sharp Reverse Line Movement detected on {away_abbrev} (+{odds_data['away_steam']['delta_prob']*100:.1f}%)")

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
                
                # Phase 6 Spreads
                'home_spread_line': odds_data.get('home_spread_line', -1.5),
                'away_spread_line': odds_data.get('away_spread_line', 1.5),
                'home_spread_odds': odds_data.get('home_spread_odds', 2.5),
                'away_spread_odds': odds_data.get('away_spread_odds', 1.5),
                'ev_home_spread': ev_home_spread,
                'ev_away_spread': ev_away_spread,
                'kelly_home_spread': locals().get('kelly_home_spread', 0),
                'kelly_away_spread': locals().get('kelly_away_spread', 0),
                
                # Phase 9 Goalies & Injuries
                'home_goalie': home_goalie,
                'away_goalie': away_goalie,
                'home_injury_penalty': home_injury_penalty,
                'away_injury_penalty': away_injury_penalty,
                
                # Phase 10 Kelly Sizing
                'kelly_ml': kelly_ml,
                'kelly_over': kelly_over,
                'kelly_under': kelly_under,
                
                # Phase 10 Special Teams & Travel
                'home_st_disparity': home_st_disparity,
                'away_st_disparity': away_st_disparity,
                'home_rest_boost': home_rest_boost,
                
                # Phase 5 Steam Tracking
                'home_steam': odds_data.get('home_steam', {}),
                'away_steam': odds_data.get('away_steam', {}),
                
                'data_source': "Odds API" if odds_data.get('is_real_data') else "Mocked"
            })
            
        return results

if __name__ == "__main__":
    print("Welcome to the Pro NHL Predictor System")
    print("-" * 40)
    app = ProfessionalNHLPredictor()
    app.run_daily_predictions()
