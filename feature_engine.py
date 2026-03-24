import pandas as pd

class FeatureEngine:
    def __init__(self):
        pass
        
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
            
            # Phase 10: Travel & Rest Calculus
            rest_days = rest_days_dict.get(team_id, 2) # Default assumption of 2 days rest if not found
            is_b2b = 1 if rest_days == 0 else 0

            # Feature Engineering Merge
            team_stats = {
                'team_id': team_id,
                'team_name': full_name,
                'win_pct': wins / games_played if games_played > 0 else 0,
                'gf_pg': goals_for / games_played if games_played > 0 else 0,
                'ga_pg': goals_against / games_played if games_played > 0 else 0,
                'goal_diff_pg': (goals_for - goals_against) / games_played if games_played > 0 else 0,
                'l10_win_pct': l10_wins / 10.0,
                'streak_momentum': streak_momentum,
                # Advanced stats
                'pp_pct': adv_data.get('powerPlayPct', 0.20), # fallback average 20%
                'pk_pct': adv_data.get('penaltyKillPct', 0.80), # fallback average 80%
                'faceoff_pct': adv_data.get('faceoffWinPct', 0.50), # fallback average 50%
                'shots_for_pg': adv_data.get('shotsForPerGame', 30.0),
                'shots_against_pg': adv_data.get('shotsAgainstPerGame', 30.0),
                
                # Phase 9: Injury Tracking Penalty
                # Reduce the team's historical Expected Goals by the exact mathematical % missing due to scratched players
                'xg_for_pg': mp_data.get('xg_for_pg', 3.0) * (1.0 - injury_impacts.get(team_id, 0.0)),
                'injury_penalty_pct': injury_impacts.get(team_id, 0.0),
                
                'xg_against_pg': mp_data.get('xg_against_pg', 3.0),
                
                # Phase 9: Individual Goalie Override
                # If we scraped a confirmed starter, use their specific SV% instead of the team's rolling average.
                # This explicitly punishes a team resting their starter, or rewards an elite Vezina goalie.
                'sv_pct': starting_goalies.get(team_id, {}).get('sv_pct') or mp_data.get('sv_pct', 0.900),
                'starting_goalie': starting_goalies.get(team_id, {}).get('name', 'Team Average'),
                
                'is_b2b': is_b2b,
                'rest_days': rest_days,
                
                # New Phase 6 Ultra-Deep Metrics
                'cf_pct': mp_data.get('cf_pct', 0.5),
                'ff_pct': mp_data.get('ff_pct', 0.5),
                'hd_shots_for': mp_data.get('hd_shots_for', 5.0),
                'hd_shots_against': mp_data.get('hd_shots_against', 5.0),
                'hd_xg_for': mp_data.get('hd_xg_for', 1.0),
                'hd_xg_against': mp_data.get('hd_xg_against', 1.0),
                'sva_xg_for': mp_data.get('sva_xg_for', 3.0),
                'sva_xg_against': mp_data.get('sva_xg_against', 3.0),
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
            'away_rest_days': away_stats['rest_days']
        }
        
        return pd.DataFrame([features])
