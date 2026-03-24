import pandas as pd
import requests
import os
import io

class HistoricalDataBuilder:
    def __init__(self):
        self.csv_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
        self.output_file = "historical_training_data.csv"

    def download_and_process(self):
        print("Downloading historical NHL data from MoneyPuck (This may take a minute...)")
        
        # Download with headers to bypass basic blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(self.csv_url, headers=headers)
        response.raise_for_status()
        
        print("Data downloaded. Processing raw CSV...")
        
        # Read the csv from the response content
        df = pd.read_csv(io.StringIO(response.text))
        
        # Filter for 'all' situations, not just 5v5
        df = df[df['situation'] == 'all'].copy()
        
        # We only want recent seasons to keep the model relevant (e.g., last 5 years)
        # MoneyPuck seasons are like 2023 for the 2023-2024 season
        recent_seasons = sorted(df['season'].unique())[-5:]
        df = df[df['season'].isin(recent_seasons)].copy()
        print(f"Filtered to the last 5 seasons: {recent_seasons}")

        # Basic outcomes
        df['win'] = (df['goalsFor'] > df['goalsAgainst']).astype(int)
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values(by=['season', 'playerTeam', 'gameDate'])

        print("Calculating point-in-time rolling averages for ML features...")
        
        # Calculate Rolling Averages (Last 10 games) per team per season
        def calculate_rolling(group):
            # Shift by 1 so the game itself isn't included in the predict features
            rolling_cols = ['win', 'goalsFor', 'goalsAgainst', 
                            'shotsOnGoalFor', 'shotsOnGoalAgainst', 
                            'faceOffsWonFor', 'faceOffsWonAgainst',
                            'xGoalsFor', 'xGoalsAgainst', 'savedShotsOnGoalAgainst']
            
            rolling = group[rolling_cols].rolling(window=10, min_periods=1).mean().shift(1)
            
            group['l10_win_pct'] = rolling['win']
            group['gf_pg'] = rolling['goalsFor']
            group['ga_pg'] = rolling['goalsAgainst']
            group['shots_for_pg'] = rolling['shotsOnGoalFor']
            group['shots_against_pg'] = rolling['shotsOnGoalAgainst']
            
            # New Advanced Metrics (Phase 4)
            group['xg_for_pg'] = rolling['xGoalsFor']
            group['xg_against_pg'] = rolling['xGoalsAgainst']
            group['sv_pct'] = (rolling['savedShotsOnGoalAgainst'] / (rolling['shotsOnGoalAgainst'] + 0.001)).fillna(0.900)
            
            # Approximate Faceoff %
            f_won = rolling['faceOffsWonFor']
            f_lost = rolling['faceOffsWonAgainst']
            group['faceoff_pct'] = f_won / (f_won + f_lost + 0.001)
            
            # Schedule Fatigue (B2B)
            group['days_rest'] = group['gameDate'].diff().dt.days
            group['is_b2b'] = (group['days_rest'] <= 1).astype(int)
            
            # Cumulative win percentage for the whole season up to this point
            cum_games = group.reset_index().index
            cum_wins = group['win'].cumsum().shift(1)
            # Avoid division by zero
            # Where cum_games is 0, we'll just set win_pct to 0.5 as a neutral prior
            group['win_pct'] = (cum_wins / cum_games).fillna(0.5)
            
            return group

        df = df.groupby(['season', 'playerTeam'], group_keys=False).apply(calculate_rolling)
        
        # Drop rows where we couldn't calculate rolling averages (e.g., first game of season)
        # or fill them with neutral values. We will fill NaNs for the very first game.
        df.fillna({
            'l10_win_pct': 0.5,
            'gf_pg': 3.0,
            'ga_pg': 3.0,
            'shots_for_pg': 30.0,
            'shots_against_pg': 30.0,
            'faceoff_pct': 0.5,
            'win_pct': 0.5,
            'xg_for_pg': 3.0,
            'xg_against_pg': 3.0,
            'sv_pct': 0.900,
            'is_b2b': 0
        }, inplace=True)

        print("Restructuring into Matchup pairs (Home vs Away)...")
        # Now we need to merge the home team rows with the away team rows for the same game
        home_games = df[df['home_or_away'] == 'HOME'].copy()
        away_games = df[df['home_or_away'] == 'AWAY'].copy()
        
        # Rename columns to distinguish home and away features
        feature_cols = ['win_pct', 'gf_pg', 'ga_pg', 'l10_win_pct', 'shots_for_pg', 'shots_against_pg', 
                        'faceoff_pct', 'xg_for_pg', 'xg_against_pg', 'sv_pct', 'is_b2b']
        
        home_features = home_games[['gameId', 'playerTeam', 'win'] + feature_cols].rename(
            columns={col: f'home_{col}' for col in feature_cols}
        )
        home_features.rename(columns={'playerTeam': 'home_team', 'win': 'home_win'}, inplace=True)
        
        away_features = away_games[['gameId', 'playerTeam'] + feature_cols].rename(
            columns={col: f'away_{col}' for col in feature_cols}
        )
        away_features.rename(columns={'playerTeam': 'away_team'}, inplace=True)
        
        # Merge on gameId
        matchups = pd.merge(home_features, away_features, on='gameId')
        
        # We also need shot differentials to match our real predictor
        matchups['home_shots_diff'] = matchups['home_shots_for_pg'] - matchups['home_shots_against_pg']
        matchups['away_shots_diff'] = matchups['away_shots_for_pg'] - matchups['away_shots_against_pg']

        # Save to CSV
        matchups.to_csv(self.output_file, index=False)
        print(f"Successfully generated {len(matchups)} historical training samples!")
        print(f"Saved to {os.path.abspath(self.output_file)}")

if __name__ == "__main__":
    builder = HistoricalDataBuilder()
    builder.download_and_process()
