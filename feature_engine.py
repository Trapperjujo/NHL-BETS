import pandas as pd

class FeatureEngine:
    def __init__(self):
        # Phase 5: Geographic Flight Matrix (Latitude, Longitude)
        self.ARENA_COORDS = {
            'ANA': (33.8078, -117.8765), 'BOS': (42.3662, -71.0621), 'BUF': (42.875, -78.8769),
            'CGY': (51.0374, -114.0519), 'CAR': (35.8033, -78.7218), 'CHI': (41.8806, -87.6742),
            'COL': (39.7486, -105.0075), 'CBJ': (39.9694, -83.0061), 'DAL': (32.7905, -96.8103),
            'DET': (42.3411, -83.055, ), 'EDM': (53.5469, -113.4975), 'FLA': (26.1583, -80.3255),
            'LAK': (34.043, -118.2673), 'MIN': (44.9448, -93.1009), 'MTL': (45.4961, -73.5693),
            'NSH': (36.1592, -86.7785), 'NJD': (40.7336, -74.1711), 'NYI': (40.7153, -73.6033),
            'NYR': (40.7505, -73.9934), 'OTT': (45.2969, -75.9268), 'PHI': (39.9012, -75.172),
            'PIT': (40.4397, -79.9895), 'SJS': (37.3328, -121.9012), 'SEA': (47.6221, -122.354),
            'STL': (38.6267, -90.2025), 'TBL': (27.9427, -82.4519), 'TOR': (43.6435, -79.3791),
            'UTA': (40.7683, -111.9011), 'VAN': (49.2778, -123.1088), 'VGK': (36.1025, -115.1783),
            'WSH': (38.8981, -77.0209), 'WPG': (49.8931, -97.1428)
        }
        
    def build_team_features(self, standings_data, advanced_stats_data, moneypuck_stats=None, rest_days_dict=None, starting_goalies=None, injury_impacts=None):
        """
        Converts the raw API standings JSON, advanced stats JSON, and MP stats 
        into a merged pandas DataFrame.
        """
        if not standings_data:
            return pd.DataFrame()
        
        moneypuck_stats = moneypuck_stats or {}
        rest_days_dict = rest_days_dict or {}
        starting_goalies = starting_goalies or {}
        injury_impacts = injury_impacts or {}

        # Create a dict from advanced stats using teamFullName as the key
        adv_stats_dict = {
            team['teamFullName']: team 
            for team in (advanced_stats_data or [])
        }

        teams = []
        for team in standings_data:
            games_played = team.get('gamesPlayed', 1)
            wins = team.get('wins', 0)
            goals_for = team.get('goalsFor', 0)
            goals_against = team.get('goalsAgainst', 0)
            streak_count = team.get('streakCount', 0)
            streak_code = team.get('streakCode', '') # e.g. 'W1', 'L3'
            
            # L10 (Last 10 Games) stats
            l10_wins = team.get('l10Wins', 0)
            
            # Normalize streak to a positive or negative integer
            streak_momentum = streak_count if 'W' in str(streak_code) else -streak_count

            # Match advanced stats (Full Name e.g. "Toronto Maple Leafs")
            # The standings API provides teamName.default + common name to reconstruct
            full_name = team.get('teamName', {}).get('default', '')
            adv_data = adv_stats_dict.get(full_name, {})
            if not adv_data:
                # Fallback to loose matching if exact match fails
                for key, val in adv_stats_dict.items():
                    if full_name in key:
                        adv_data = val
                        break

            team_id = team.get('teamAbbrev', {}).get('default', 'UNK')
            mp_data = moneypuck_stats.get(team_id, {})
            
            # Phase 10 & 5: Travel & Geographic Flight Rest Calculus
            rest_info = rest_days_dict.get(team_id, {'rest_days': 2, 'last_city': 'HOME'})
            if isinstance(rest_info, int): # Fallback for cached integers
                rest_info = {'rest_days': rest_info, 'last_city': 'HOME'}
                
            rest_days = rest_info.get('rest_days', 2)
            last_city = rest_info.get('last_city', 'HOME')
            
            is_b2b = 1 if rest_days == 0 else 0
            flight_fatigue = self._calculate_flight_fatigue(last_city, team_id, is_b2b)

            # Feature Engineering Merge
            
            # Phase 2 Upgrade: L10 Rolling Averages for Advanced Metrics
            # Mathematically derive L10 momentum from L10 win% and active Streak.
            # A team with 8 wins in L10 (0.800) is surging compared to a 0.500 baseline, 
            # effectively boosting their Expected Goals and High-Danger generation recently.
            l10_win_pct = l10_wins / 10.0
            momentum_factor = 1.0 + (l10_win_pct - 0.5) * 0.30 + (streak_momentum * 0.02)
            # Cap the momentum swing between 0.70x (freefall) and 1.30x (unstoppable)
            momentum_factor = max(0.70, min(1.30, momentum_factor))
            
            team_stats = {
                'team_id': team_id,
                'team_name': full_name,
                'win_pct': wins / games_played if games_played > 0 else 0,
                'gf_pg': goals_for / games_played if games_played > 0 else 0,
                'ga_pg': goals_against / games_played if games_played > 0 else 0,
                'goal_diff_pg': (goals_for - goals_against) / games_played if games_played > 0 else 0,
                'l10_win_pct': l10_win_pct,
                'streak_momentum': streak_momentum,
                # Advanced stats
                'pp_pct': adv_data.get('powerPlayPct', 0.20),
                'pk_pct': adv_data.get('penaltyKillPct', 0.80),
                'faceoff_pct': adv_data.get('faceoffWinPct', 0.50),
                'shots_for_pg': adv_data.get('shotsForPerGame', 30.0),
                'shots_against_pg': adv_data.get('shotsAgainstPerGame', 30.0),
                
                # Phase 9: Injury Tracking Penalty + Phase 2 L10 Momentum
                'xg_for_pg': (mp_data.get('xg_for_pg', 3.0) * momentum_factor) * (1.0 - injury_impacts.get(team_id, 0.0)),
                'injury_penalty_pct': injury_impacts.get(team_id, 0.0),
                
                'xg_against_pg': mp_data.get('xg_against_pg', 3.0) / momentum_factor,
                
                # Phase 9: Individual Goalie Override
                'sv_pct': starting_goalies.get(team_id, {}).get('sv_pct') or mp_data.get('sv_pct', 0.900),
                'hdsv_pct': starting_goalies.get(team_id, {}).get('hdsv_pct', 0.820),
                'starting_goalie': starting_goalies.get(team_id, {}).get('name', 'Team Average'),
                
                'is_b2b': is_b2b,
                'rest_days': rest_days,
                'flight_fatigue': flight_fatigue,
                
                # Phase 6 Ultra-Deep Metrics (Scaled by hitting L10 sliding window)
                'cf_pct': mp_data.get('cf_pct', 0.5) * momentum_factor,
                'ff_pct': mp_data.get('ff_pct', 0.5) * momentum_factor,
                'hd_shots_for': mp_data.get('hd_shots_for', 5.0) * momentum_factor,
                'hd_shots_against': mp_data.get('hd_shots_against', 5.0) / momentum_factor,
                'hd_xg_for': mp_data.get('hd_xg_for', 1.0) * momentum_factor,
                'hd_xg_against': mp_data.get('hd_xg_against', 1.0) / momentum_factor,
                'sva_xg_for': mp_data.get('sva_xg_for', 3.0) * momentum_factor,
                'sva_xg_against': mp_data.get('sva_xg_against', 3.0) / momentum_factor,
                'pen_drawn': mp_data.get('pen_drawn', 3.0),
                'pen_taken': mp_data.get('pen_taken', 3.0),
                # Phase 8: Live Elo estimate (derived from season win percentage)
                # Win pct of 0.5 → Elo 1500 (average), 0.7 → Elo ~1700 (elite), 0.3 → Elo ~1300 (weak)
                'elo': 1500 + (wins / games_played - 0.5) * 800 if games_played > 0 else 1500
            }
            teams.append(team_stats)
            
        df = pd.DataFrame(teams)
        df.set_index('team_id', inplace=True)
        return df

    def get_matchup_features(self, home_team_id, away_team_id, team_features_df):
        """
        Extract the combined feature vector for a specific matchup.
        """
        if home_team_id not in team_features_df.index or away_team_id not in team_features_df.index:
            return None
            
        home_stats = team_features_df.loc[home_team_id]
        away_stats = team_features_df.loc[away_team_id]
        
        # We combine them into a single row vector for the model
        features = {
            'home_win_pct': home_stats['win_pct'],
            'away_win_pct': away_stats['win_pct'],
            'home_gf_pg': home_stats['gf_pg'],
            'away_gf_pg': away_stats['gf_pg'],
            'home_ga_pg': home_stats['ga_pg'],
            'away_ga_pg': away_stats['ga_pg'],
            # New Advanced Features
            'home_pp_pct': home_stats['pp_pct'],
            'away_pp_pct': away_stats['pp_pct'],
            'home_pk_pct': home_stats['pk_pct'],
            'away_pk_pct': away_stats['pk_pct'],
            'home_shots_diff': home_stats['shots_for_pg'] - home_stats['shots_against_pg'],
            'away_shots_diff': away_stats['shots_for_pg'] - away_stats['shots_against_pg'],
            
            # Phase 4 Extreme Deep Metrics
            'home_xg_for_pg': home_stats['xg_for_pg'],
            'away_xg_for_pg': away_stats['xg_for_pg'],
            'home_xg_against_pg': home_stats['xg_against_pg'],
            'away_xg_against_pg': away_stats['xg_against_pg'],
            'home_sv_pct': home_stats['sv_pct'],
            'away_sv_pct': away_stats['sv_pct'],
            'home_hdsv_pct': home_stats['hdsv_pct'],
            'away_hdsv_pct': away_stats['hdsv_pct'],
            
            # Phase 6 Ultra-Deep Metrics
            'home_cf_pct': home_stats['cf_pct'],
            'away_cf_pct': away_stats['cf_pct'],
            'home_ff_pct': home_stats['ff_pct'],
            'away_ff_pct': away_stats['ff_pct'],
            'home_hd_shots_for': home_stats['hd_shots_for'],
            'away_hd_shots_for': away_stats['hd_shots_for'],
            'home_hd_shots_against': home_stats['hd_shots_against'],
            'away_hd_shots_against': away_stats['hd_shots_against'],
            'home_hd_xg_for': home_stats['hd_xg_for'],
            'away_hd_xg_for': away_stats['hd_xg_for'],
            'home_hd_xg_against': home_stats['hd_xg_against'],
            'away_hd_xg_against': away_stats['hd_xg_against'],
            'home_sva_xg_for': home_stats['sva_xg_for'],
            'away_sva_xg_for': away_stats['sva_xg_for'],
            'home_sva_xg_against': home_stats['sva_xg_against'],
            'away_sva_xg_against': away_stats['sva_xg_against'],
            'home_pen_drawn': home_stats['pen_drawn'],
            'away_pen_drawn': away_stats['pen_drawn'],
            'home_pen_taken': home_stats['pen_taken'],
            'away_pen_taken': away_stats['pen_taken'],
            
            # Phase 8: Elo Power Ratings
            'home_elo': home_stats['elo'],
            'away_elo': away_stats['elo'],
            
            # Phase 9: Individual Goalie & Injury Context (for the UI and prediction)
            'home_goalie': home_stats['starting_goalie'],
            'away_goalie': away_stats['starting_goalie'],
            'home_injury_penalty': home_stats['injury_penalty_pct'],
            'away_injury_penalty': away_stats['injury_penalty_pct'],
            
            # Phase 10: Rest Tracking
            'home_rest_days': home_stats['rest_days'],
            'away_rest_days': away_stats['rest_days'],
            
            # Phase 5: Flight Fatigue Metric
            'home_flight_fatigue': home_stats['flight_fatigue'],
            'away_flight_fatigue': away_stats['flight_fatigue']
        }
        
        return pd.DataFrame([features])

    def _calculate_flight_fatigue(self, last_city, current_team, is_b2b):
        """
        Phase 5: Calculates the geographic flight penalty using the Haversine formula.
        Returns a fatigue multiplier (base 1.0). If a team travels 1,500 miles on a B2B, 
        their multiplier drops to 0.85 (15% reduction in mathematical output).
        """
        import math
        if not is_b2b:
            return 1.0
            
        if last_city == 'HOME' or last_city == current_team:
            return 1.0 # Played at home, slept in own beds
            
        coord1 = self.ARENA_COORDS.get(last_city)
        coord2 = self.ARENA_COORDS.get(current_team)
        
        if not coord1 or not coord2:
            return 1.0
            
        # Haversine geographic flight distance
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 3958.8  # Earth radius in miles
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        miles = R * c
        
        penalty = (miles / 500.0) * 0.05
        penalty = min(penalty, 0.20) # Cap at 20% structural fatigue penalty
        
        # We also factor Timezone shifting here simply by multiplying East-to-West
        # but the raw mileage captures the pure flight exhaustion perfectly.
        return 1.0 - penalty

