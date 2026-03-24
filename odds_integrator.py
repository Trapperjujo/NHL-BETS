import random
import os
import requests

class OddsIntegrator:
    def __init__(self):
        # Base realistic odds boundaries (Decimal odds)
        self.min_odds = 1.15 # Massive favorite
        self.max_odds = 4.50 # Massive underdog
        self.real_odds_cache = None

    def fetch_live_odds(self, home_team_name, away_team_name, home_prob_estimate):
        """
        Attempts to fetch real sportsbook odds if an API key is present.
        Falls back to a realistic mock generator if not.
        """
        api_key = os.environ.get('ODDS_API_KEY')
        if api_key:
            real_odds = self._fetch_real_odds(api_key, home_team_name, away_team_name)
            if real_odds:
                return real_odds
                
        # Fallback to mock
        return self.mock_fetch_live_odds(home_prob_estimate)

    def _fetch_real_odds(self, api_key, home_team, away_team):
        try:
            if not self.real_odds_cache:
                url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?regions=us&markets=h2h&oddsFormat=decimal&apiKey={api_key}"
                resp = requests.get(url)
                resp.raise_for_status()
                self.real_odds_cache = resp.json()
                
            for game in self.real_odds_cache:
                # The Odds API uses full names (e.g., "Toronto Maple Leafs")
                if home_team in game['home_team'] and away_team in game['away_team']:
                    # Extract the first bookmaker's odds (e.g. DraftKings)
                    if game.get('bookmakers'):
                        markets = game['bookmakers'][0]['markets'][0]['outcomes']
                        home_odds = next(o['price'] for o in markets if o['name'] == game['home_team'])
                        away_odds = next(o['price'] for o in markets if o['name'] == game['away_team'])
                        return {
                            'home_odds': home_odds,
                            'away_odds': away_odds,
                            'is_real_data': True
                        }
        except Exception as e:
            print(f"Failed to fetch real odds: {e}")
        return None

    def mock_fetch_live_odds(self, home_team_win_prob_estimate):
        """
        Simulates fetching live sportsbook odds.
        In reality, you would connect to The Odds API or a sportsbook.
        
        We mock the bookmaker's implied probability using roughly the true probability
        plus some "juice" (the bookmaker's edge) and randomness.
        """
        # Bookmaker implied probability (true + edge)
        juice = 0.045 # Roughly 4.5% vigorish
        book_implied_home = home_team_win_prob_estimate + random.uniform(-0.05, 0.05)
        
        # Ensure it stays within reasonable bounds
        book_implied_home = max(0.1, min(0.9, book_implied_home))
        book_implied_away = 1.0 - book_implied_home + juice
        book_implied_home += (juice / 2)
        
        # Convert implied probability to decimal odds
        home_odds = round(1.0 / book_implied_home, 2)
        away_odds = round(1.0 / book_implied_away, 2)
        
        return {
            'home_odds': min(self.max_odds, max(self.min_odds, home_odds)),
            'away_odds': min(self.max_odds, max(self.min_odds, away_odds)),
            'is_real_data': False
        }

    def calculate_ev(self, model_prob, decimal_odds):
        """
        Calculates Expected Value (+EV) of a $100 bet.
        Formula: (Probability of Winning x Amount Won per Bet) - (Probability of Losing x Amount Lost)
        Amount Won = 100 * (Odds - 1)
        """
        prob_win = model_prob
        prob_lose = 1.0 - prob_win
        
        amount_won = 100 * (decimal_odds - 1)
        amount_lost = 100
        
        expected_value = (prob_win * amount_won) - (prob_lose * amount_lost)
        return round(expected_value, 2)
