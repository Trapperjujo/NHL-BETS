import random
import os
import requests
from dotenv import load_dotenv

load_dotenv()

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
                url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?regions=us&markets=h2h,totals&oddsFormat=decimal&apiKey={api_key}"
                resp = requests.get(url)
                resp.raise_for_status()
                self.real_odds_cache = resp.json()
                
            home_clean = home_team.replace('.', '')
            away_clean = away_team.replace('.', '')
            
            for game in self.real_odds_cache:
                api_home_clean = game['home_team'].replace('.', '')
                api_away_clean = game['away_team'].replace('.', '')
                
                # The Odds API uses full names (e.g., "Toronto Maple Leafs")
                if home_clean in api_home_clean and away_clean in api_away_clean:
                    if game.get('bookmakers'):
                        markets = game['bookmakers'][0]['markets']
                        
                        home_odds, away_odds = None, None
                        over_odds, under_odds, o_u_line = None, None, 6.5
                        
                        for m in markets:
                            if m['key'] == 'h2h':
                                home_odds = next((o['price'] for o in m['outcomes'] if o['name'] == game['home_team']), None)
                                away_odds = next((o['price'] for o in m['outcomes'] if o['name'] == game['away_team']), None)
                            elif m['key'] == 'totals':
                                # Totals outcomes are named 'Over' and 'Under'. They also provide 'point' (the line)
                                over_outcome = next((o for o in m['outcomes'] if o['name'] == 'Over'), None)
                                under_outcome = next((o for o in m['outcomes'] if o['name'] == 'Under'), None)
                                if over_outcome and under_outcome:
                                    over_odds = over_outcome['price']
                                    under_odds = under_outcome['price']
                                    o_u_line = over_outcome.get('point', 6.5)
                        
                        return {
                            'home_odds': home_odds,
                            'away_odds': away_odds,
                            'over_odds': over_odds,
                            'under_odds': under_odds,
                            'o_u_line': o_u_line,
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
        
        # Mock Totals (O/U) roughly centered around 6.5 goals in the NHL
        o_u_line = random.choice([5.5, 6.0, 6.5])
        over_odds = round(random.uniform(1.85, 2.05), 2)
        under_odds = round(random.uniform(1.85, 2.05), 2)
        
        return {
            'home_odds': min(self.max_odds, max(self.min_odds, home_odds)),
            'away_odds': min(self.max_odds, max(self.min_odds, away_odds)),
            'over_odds': over_odds,
            'under_odds': under_odds,
            'o_u_line': o_u_line,
            'is_real_data': False
        }

    def calculate_ev(self, model_prob, decimal_odds):
        """
        Calculates Expected Value (+EV) of a $100 bet.
        Formula: (Probability of Winning x Amount Won per Bet) - (Probability of Losing x Amount Lost)
        Amount Won = 100 * (Odds - 1)
        """
        if decimal_odds is None:
            return 0.0

        prob_win = model_prob
        prob_lose = 1.0 - prob_win
        
        amount_won = 100 * (decimal_odds - 1)
        amount_lost = 100
        
        expected_value = (prob_win * amount_won) - (prob_lose * amount_lost)
        return round(expected_value, 2)

    def calculate_kelly_criterion(self, model_prob, decimal_odds, bankroll, fraction=0.25):
        """
        Phase 10: Advanced Bankroll Management
        Calculates the mathematically optimal wager size using the Fractional Kelly Criterion.
        f* = (bp - q) / b
        where:
        f* is the fraction of the current bankroll to wager.
        b is the net fractional odds received on the wager (decimal_odds - 1).
        p is the probability of winning (model_prob).
        q is the probability of losing (1 - p).
        
        fraction=0.25 means we use a Quarter-Kelly strategy, which is the gold standard for sports 
        betting to minimize variance intuitively and avoid catastrophic drawdowns.
        """
        if decimal_odds <= 1.0:
            return 0.0
            
        b = decimal_odds - 1.0
        p = model_prob
        q = 1.0 - p
        
        kelly_fraction = (b * p - q) / b
        
        # If the expected value is negative, the Kelly formula outputs <= 0 (Do not bet)
        if kelly_fraction <= 0:
            return 0.0
            
        # Apply the fractional safety net (e.g. Quarter Kelly)
        safe_fraction = kelly_fraction * fraction
        
        # Cap a singular max bet at 5% of bankroll to protect against statistical outliers
        safe_fraction = min(safe_fraction, 0.05)
        
        return round(safe_fraction, 5)
