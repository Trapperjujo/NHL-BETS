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
