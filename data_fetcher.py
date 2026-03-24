import requests
import datetime
import json

class NHLDataFetcher:
    def __init__(self):
        self.standings_url = "https://api-web.nhle.com/v1/standings/now"
        self.schedule_url = "https://api-web.nhle.com/v1/schedule/now"
        self.stats_url = "https://api.nhle.com/stats/rest/en/team/summary"

    def fetch_advanced_stats(self):
        """Fetch advanced metrics like PP%, PK%, Faceoffs from the NHL Stats API."""
        try:
            response = requests.get(self.stats_url)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.RequestException as e:
            print(f"Error fetching advanced stats: {e}")
            return []

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
                    'sv_pct': row['savedShotsOnGoalAgainst'] / (row['shotsOnGoalAgainst'] + 0.001)
                }
            return stats
        except Exception as e:
            print(f"Error fetching live MoneyPuck stats: {e}")
            return {}
