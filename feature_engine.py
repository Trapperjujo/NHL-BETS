import pandas as pd

class FeatureEngine:
    def __init__(self):
        pass
        
    def build_team_features(self, standings_data, advanced_stats_data):
        """
        Converts the raw API standings JSON and advanced stats JSON 
        into a merged pandas DataFrame.
        """
        if not standings_data:
            return pd.DataFrame()

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

            # Feature Engineering Merge
            team_stats = {
                'team_id': team.get('teamAbbrev', {}).get('default', 'UNK'),
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
                'shots_against_pg': adv_data.get('shotsAgainstPerGame', 30.0)
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
            'home_l10_win_pct': home_stats['l10_win_pct'],
            'away_l10_win_pct': away_stats['l10_win_pct'],
            'home_streak': home_stats['streak_momentum'],
            'away_streak': away_stats['streak_momentum'],
            # New Advanced Features
            'home_pp_pct': home_stats['pp_pct'],
            'away_pp_pct': away_stats['pp_pct'],
            'home_pk_pct': home_stats['pk_pct'],
            'away_pk_pct': away_stats['pk_pct'],
            'home_shots_diff': home_stats['shots_for_pg'] - home_stats['shots_against_pg'],
            'away_shots_diff': away_stats['shots_for_pg'] - away_stats['shots_against_pg']
        }
        
        return pd.DataFrame([features])
