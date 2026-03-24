import requests
import datetime
import json

class NHLDataFetcher:
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1"
        self.moneypuck_base = "https://moneypuck.com/moneypuck/playerData"
        self.espn_scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
        
        # ESPN uses slightly different team abbreviations than NHL.com
        self.espn_to_nhl_map = {
            'TB': 'TBL', 'SJ': 'SJS', 'LA': 'LAK', 'NJ': 'NJD', 'WSH': 'WSH',
            'BOS': 'BOS', 'TOR': 'TOR', 'MTL': 'MTL', 'CAR': 'CAR', 'DET': 'DET',
            'OTT': 'OTT', 'FLA': 'FLA', 'SEA': 'SEA', 'NYI': 'NYI', 'CHI': 'CHI',
            'PHI': 'PHI', 'CBJ': 'CBJ', 'PIT': 'PIT', 'COL': 'COL', 'MIN': 'MIN',
            'STL': 'STL', 'NSH': 'NSH', 'DAL': 'DAL', 'WPG': 'WPG', 'VGK': 'VGK',
            'CGY': 'CGY', 'UTA': 'UTA', 'EDM': 'EDM', 'VAN': 'VAN', 'ANA': 'ANA',
            'NYR': 'NYR', 'BUF': 'BUF'
        }
        self.standings_url = f"{self.base_url}/standings/now"
        self.schedule_url = f"{self.base_url}/schedule/now"
        self.stats_url = "https://api.nhle.com/stats/rest/en/team/summary"

    def fetch_advanced_stats(self):
        """Fetch advanced metrics like PP%, PK%, Faceoffs from the NHL Stats API."""
        try:
            response = requests.get(self.stats_url)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.RequestException as e:
            print(f"Failed to fetch MoneyPuck 5v5 deep metrics: {e}")
            return None

    def fetch_starting_goalies(self):
        """
        Phase 9 Upgrade: Individual Player Tracking
        - Hits ESPN API to identify confirmed/expected starting goalies tonight.
        - Hits MoneyPuck to pull individual goalie SV% and GSAx (Goals Saved Above Expected).
        Returns a dict: { 'TOR': {'name': 'Anthony Stolarz', 'sv_pct': 0.925, 'gsax': 15.2} }
        """
        import io
        import pandas as pd
        goalie_data = {}
        
        try:
            # 1. Fetch Tonight's Starters from ESPN
            espn_resp = requests.get(self.espn_scoreboard_url)
            espn_resp.raise_for_status()
            events = espn_resp.json().get('events', [])
            
            starters_by_team = {}
            for event in events:
                for comp in event['competitions'][0]['competitors']:
                    team_abbrev_espn = comp['team']['abbreviation']
                    team_abbrev_nhl = self.espn_to_nhl_map.get(team_abbrev_espn, team_abbrev_espn)
                    
                    probables = comp.get('probables', [])
                    if probables:
                        # Grab the first probable goalie
                        goalie_name = probables[0]['athlete']['displayName']
                        status = probables[0]['status']['name'] # 'Confirmed' or 'Expected'
                        starters_by_team[team_abbrev_nhl] = {'name': goalie_name, 'status': status}
            
            if not starters_by_team:
                print("ESPN returned no probable goalies for tonight. Falling back to team averages.")
                return {}
                
            # 2. Fetch MoneyPuck Individual Goalie Stats
            headers = {'User-Agent': 'Mozilla/5.0'}
            # Note: Hardcoding '2024' or fetching 'current' based on season logic. 2024=2024-25 season.
            url = f"{self.moneypuck_base}/seasonSummary/2024/regular/goalies.csv" 
            mp_resp = requests.get(url, headers=headers)
            mp_resp.raise_for_status()
            
            df = pd.read_csv(io.StringIO(mp_resp.text))
            
            # Filter to 'all' situations
            df = df[df['situation'] == 'all']
            
            # Map ESPN starer names to MoneyPuck stat lines
            for team, starter in starters_by_team.items():
                name = starter['name']
                # Search MoneyPuck CSV for this goalie
                # Handles slight naming variations (e.g., 'Sam Montembeault' vs 'Samuel Montembeault')
                goalie_row = df[df['name'].str.contains(name.split()[-1], case=False, na=False)]
                
                if not goalie_row.empty:
                    # If multiple result (e.g. Sebastian Aho), just take the one with most shots faced
                    goalie_row = goalie_row.sort_values(by='shotsOnGoal', ascending=False).iloc[0]
                    
                    sv_pct = goalie_row['savedShotsOnGoal'] / (goalie_row['shotsOnGoal'] + 0.001)
                    gsax = goalie_row['savedChancesAboveExpected']
                    
                    goalie_data[team] = {
                        'name': goalie_row['name'],
                        'status': starter['status'],
                        'sv_pct': sv_pct,
                        'gsax': gsax
                    }
                else:
                    # Goalie not found (e.g., AHL call up)
                    goalie_data[team] = {'name': name, 'status': starter['status'], 'sv_pct': None, 'gsax': None}
                    
            print(f"Successfully scraped individual analytics for {len(goalie_data)} starting goalies.")
            return goalie_data
            
        except Exception as e:
            print(f"Failed to fetch individual starting goalie data: {e}")
            return {}

    def fetch_injury_impacts(self, team_abbrevs):
        """
        Phase 9 Upgrade: Individual Player Tracking
        - Scrapes ESPN to find injured/scratched players for tonight's teams.
        - Cross-references MoneyPuck skaters.csv to calculate the exact % of 
          Expected Goals (xG) the team is missing due to these injuries.
        Returns a dict: { 'EDM': 0.15 }  (meaning Edmonton is missing 15% of their xG firepower)
        """
        import pandas as pd
        import io
        
        impacts = {}
        try:
            # 1. Download MoneyPuck Skater Data (all skaters, season aggregate)
            headers = {'User-Agent': 'Mozilla/5.0'}
            # Note: Hardcoding '2024' or fetching 'current' based on season logic. 2024=2024-25 season.
            url = f"{self.moneypuck_base}/seasonSummary/2024/regular/skaters.csv" 
            mp_resp = requests.get(url, headers=headers)
            mp_resp.raise_for_status()
            
            df = pd.read_csv(io.StringIO(mp_resp.text))
            df = df[df['situation'] == 'all']
            
            # Map Team Abbrevs back to ESPN format for the API call
            nhl_to_espn_map = {v: k for k, v in self.espn_to_nhl_map.items()}
            
            for team_nhl in team_abbrevs:
                team_espn = nhl_to_espn_map.get(team_nhl, team_nhl).lower()
                
                # Fetch ESPN Roster/Injuries
                espn_resp = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_espn}/roster")
                if espn_resp.status_code != 200:
                    impacts[team_nhl] = 0.0
                    continue
                    
                roster_data = espn_resp.json()
                injured_names = []
                
                for group in roster_data.get('athletes', []):
                    for player in group.get('items', []):
                        injuries = player.get('injuries', [])
                        if injuries:
                            status = injuries[0].get('status', '').lower()
                            # Ensure we only track players who are actually missing the game
                            if status in ['out', 'injured reserve', 'day-to-day']:
                                injured_names.append(player.get('fullName'))
                
                if not injured_names:
                    impacts[team_nhl] = 0.0
                    continue
                    
                # Calculate the team's total expected goals (from MoneyPuck)
                # MoneyPuck uses their own team abbreviations, usually matches NHL.
                team_df = df[df['team'] == team_nhl]
                if team_df.empty:
                    # Some MoneyPuck team abbrevs might differ (e.g. T.B instead of TBL) - handle standard exceptions
                    mp_map = {'TBL': 'T.B', 'NJD': 'N.J', 'SJS': 'S.J', 'LAK': 'L.A'}
                    mp_team = mp_map.get(team_nhl, team_nhl)
                    team_df = df[df['team'] == mp_team]
                
                if team_df.empty:
                    impacts[team_nhl] = 0.0
                    continue
                    
                total_team_xg = team_df['I_F_xGoals'].sum()
                if total_team_xg <= 0:
                    impacts[team_nhl] = 0.0
                    continue
                    
                # Sum the missing xG
                missing_xg_total = 0.0
                for missing_player in injured_names:
                    # Fuzz match the player name
                    player_row = team_df[team_df['name'].str.contains(missing_player.split()[-1], case=False, na=False)]
                    if not player_row.empty:
                        # take highest xg if multiple matches (e.g. players traded mid season)
                        missing_xg_total += player_row['I_F_xGoals'].max()
                        
                # Provide a cap (max 40%) so a team isn't penalized 100% just because of bad data matching or massive rebuilding
                impact_pct = min(missing_xg_total / total_team_xg, 0.40) 
                impacts[team_nhl] = impact_pct
                
                if impact_pct > 0.05: # Only log major impacts
                    print(f"[{team_nhl}] Missing {impact_pct*100:.1f}% of offensive xG due to injury! (Out: {', '.join(injured_names)})")
                    
            return impacts
            
        except Exception as e:
            print(f"Failed to fetch injury impact data: {e}")
            return {}

    def fetch_current_standings(self):
        """Fetch real-time regular season standings and team stats."""
        try:
            response = requests.get(self.standings_url)
            response.raise_for_status()
            data = response.json()
            return data.get('standings', [])
        except requests.RequestException as e:
            print(f"Error fetching standings: {e}")
            return []

    def fetch_todays_schedule(self):
        """Fetch the games scheduled for today."""
        try:
            response = requests.get(self.schedule_url)
            response.raise_for_status()
            data = response.json()
            
            # The API returns 'gameWeek' which is a list of days.
            # We want to find the games for today (or the next available day with games)
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            
            for day in data.get('gameWeek', []):
                if day.get('date') == today_str:
                    return day.get('games', [])
            
            # If no games today, just grab the next available day
            if data.get('gameWeek'):
                first_day = data['gameWeek'][0]
                print(f"No games found for today ({today_str}). Showing games for {first_day.get('date')}.")
                return first_day.get('games', [])
                
            return []
            
        except requests.RequestException as e:
            print(f"Error fetching schedule: {e}")
            return []

    def fetch_tired_teams(self):
        """Returns a list of team abbreviations that played yesterday (Back-to-Back)."""
        yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/schedule/{yesterday_str}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            tired_teams = []
            for day in data.get('gameWeek', []):
                if day.get('date') == yesterday_str:
                    for game in day.get('games', []):
                        tired_teams.append(game.get('homeTeam', {}).get('abbrev'))
                        tired_teams.append(game.get('awayTeam', {}).get('abbrev'))
            return tired_teams
        except requests.RequestException:
            return []

    def fetch_moneypuck_stats(self):
        """Fetches advanced xG and SV% metrics for all teams from MoneyPuck's daily CSV."""
        import io
        import pandas as pd
        url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2024/regular/teams.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            
            # Filter for 'all' situations
            df = df[df['situation'] == 'all']
            
            stats = {}
            for _, row in df.iterrows():
                team_abbr = row['team']
                games = row['games_played']
                if games == 0: continue
                
                stats[team_abbr] = {
                    'xg_for_pg': row['xGoalsFor'] / games,
                    'xg_against_pg': row['xGoalsAgainst'] / games,
                    'sv_pct': row['savedShotsOnGoalAgainst'] / (row['shotsOnGoalAgainst'] + 0.001),
                    # Phase 6 Ultra-Deep Metrics
                    'cf_pct': row.get('corsiPercentage', 0.5),
                    'ff_pct': row.get('fenwickPercentage', 0.5),
                    'hd_shots_for': row.get('highDangerShotsFor', 5.0) / games,
                    'hd_shots_against': row.get('highDangerShotsAgainst', 5.0) / games,
                    'hd_xg_for': row.get('highDangerxGoalsFor', 1.0) / games,
                    'hd_xg_against': row.get('highDangerxGoalsAgainst', 1.0) / games,
                    'sva_xg_for': row.get('scoreVenueAdjustedxGoalsFor', 3.0) / games,
                    'sva_xg_against': row.get('scoreVenueAdjustedxGoalsAgainst', 3.0) / games,
                    'pen_drawn': row.get('penaltiesAgainst', 3.0) / games,
                    'pen_taken': row.get('penaltiesFor', 3.0) / games
                }
            return stats
        except Exception as e:
            print(f"Error fetching live MoneyPuck stats: {e}")
            return {}
