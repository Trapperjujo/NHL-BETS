import requests
import json
import math
from datetime import datetime

class PlayerPropEngine:
    """
    Phase 5: The Derivative Engine.
    Focuses specifically on Player Shots On Goal (SOG). SOG mathematically conforms closely 
    to a Poisson distribution. By scraping MoneyPuck player shot rates and comparing
    against the Odds API SOG lines, we can identify mispriced lines (especially for 
    lower-tier players).
    """
    def __init__(self, odds_api_key):
        self.odds_api_key = odds_api_key
        self.moneypuck_skaters_url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv"
        
    def fetch_player_shot_baselines(self):
        """
        Scrapes MoneyPuck and calculates the 'Expected Shots per Game' for every active player.
        """
        import pandas as pd
        import io
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(self.moneypuck_skaters_url, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            
            # Filter for 'all' situations
            df = df[df['situation'] == 'all']
            
            # We want players who have played at least 5 games to avoid small sample size variance
            df = df[df['games_played'] >= 5]
            
            player_stats = {}
            for _, row in df.iterrows():
                name = row['name']
                games = row['games_played']
                actual_shots = row.get('I_F_shotsOnGoal', 0)
                
                shots_per_game = actual_shots / games if games > 0 else 0
                
                player_stats[name] = {
                    'team': row.get('team', 'UNK'),
                    'games_played': games,
                    'shots_per_game': shots_per_game,
                    'toi_per_game': row.get('I_F_shotAttempts', 0) / 60 # Using attempts as a rough usage proxy if TOI mapping is hard
                }
            return player_stats
        except Exception as e:
            print(f"Error fetching MoneyPuck skaters: {e}")
            return {}
            
    def fetch_live_sog_lines(self):
        """
        Pulls from The Odds API for the 'player_shots_on_goal' market.
        (Note: Requires an active Odds API Key)
        """
        # Note: If no API key is provided, we will mock the data to test the algorithm logic.
        if not self.odds_api_key or self.odds_api_key == "YOUR_API_KEY_HERE":
            return self._mock_sog_lines()
            
        url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?apiKey={self.odds_api_key}&regions=us,eu&markets=player_shots_on_goal&bookmakers=pinnacle,draftkings"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return self._mock_sog_lines()
        except Exception:
            return self._mock_sog_lines()
            
    def _mock_sog_lines(self):
        """Mocks live DraftKings/Pinnacle prop lines for testing EV logic."""
        return [
            {'playerName': 'Auston Matthews', 'line': 4.5, 'over_odds': 1.95, 'under_odds': 1.85},
            {'playerName': 'Connor McDavid', 'line': 3.5, 'over_odds': 1.65, 'under_odds': 2.25},
            {'playerName': 'Sam Reinhart', 'line': 2.5, 'over_odds': 1.80, 'under_odds': 1.95}, # Often mispriced
            {'playerName': 'Zach Hyman', 'line': 3.5, 'over_odds': 2.20, 'under_odds': 1.65}
        ]

    def _poisson_probability(self, expected, target):
        """Calculates exact Poisson probability of an integer outcome."""
        return ((expected ** target) * math.exp(-expected)) / math.factorial(int(target))
        
    def calculate_sog_probabilities(self, expected_shots, line):
        """
        Uses a Poisson Distribution curve to calculate the cumulative probability 
        that a player goes OVER or UNDER their given prop line.
        """
        # Because SOG is a whole number, going OVER 2.5 means 3+ shots.
        # We calculate the probability of 0, 1, and 2 shots, and subtract from 1.0
        target = math.floor(line) # 2.5 -> 2
        
        prob_under = 0.0
        for i in range(0, target + 1):
            prob_under += self._poisson_probability(expected_shots, i)
            
        prob_over = 1.0 - prob_under
        return prob_over, prob_under

    def find_prop_edges(self):
        """
        Main orchestration method:
        1. Parse expected shots per game from MoneyPuck
        2. Parse live betting lines
        3. Match the players and calculate edge
        """
        print("Gathering Top-Down SOG Prop derivatives...")
        baselines = self.fetch_player_shot_baselines()
        live_lines = self.fetch_live_sog_lines()
        
        edges = []
        for prop in live_lines:
            name = prop['playerName']
            if name in baselines:
                expected_shots = baselines[name]['shots_per_game']
                line = prop['line']
                over_odds = prop['over_odds']
                under_odds = prop['under_odds']
                
                # SOG poisson distribution
                prob_over, prob_under = self.calculate_sog_probabilities(expected_shots, line)
                
                # Calculate True Odds based on probability
                true_over_odds = 1.0 / prob_over if prob_over > 0 else 999
                true_under_odds = 1.0 / prob_under if prob_under > 0 else 999
                
                edge_over = 0
                edge_under = 0
                
                # If market odds pay out better than true probability demands
                if over_odds > true_over_odds:
                    edge_over = (over_odds * prob_over) - 1.0
                if under_odds > true_under_odds:
                    edge_under = (under_odds * prob_under) - 1.0
                    
                if edge_over > 0.05 or edge_under > 0.05:
                    edges.append({
                        'player': name,
                        'line': line,
                        'expected_shots': round(expected_shots, 2),
                        'recommendation': 'OVER' if edge_over > edge_under else 'UNDER',
                        'edge': edge_over if edge_over > edge_under else edge_under,
                        'odds': over_odds if edge_over > edge_under else under_odds
                    })
        
        # Sort by edge strength
        edges = sorted(edges, key=lambda x: x['edge'], reverse=True)
        return edges

if __name__ == "__main__":
    engine = PlayerPropEngine("eb0d566f1f9bdaffc12ed72253d98fc4")
    edges = engine.find_prop_edges()
    print(f"Found {len(edges)} massive +EV Player Prop Bets:")
    for edge in edges:
        print(f"[{edge['player']}] -> {edge['recommendation']} {edge['line']} SOG (Expected: {edge['expected_shots']}) | Edge: +{edge['edge']*100:.1f}%")
