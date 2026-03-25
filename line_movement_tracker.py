import json
import os
import datetime

class LineMovementTracker:
    def __init__(self, cache_file="odds_cache.json"):
        self.cache_file = cache_file
        
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_cache(self, data):
        with open(self.cache_file, "w") as f:
            json.dump(data, f, indent=4)

    def log_odds(self, game_id, team_name, current_odds):
        """
        Logs the current odds. If it's the first time seeing this game today, 
        it saves it as the 'opening_odds'. Returns the delta from opening.
        """
        if current_odds is None:
            return 0.0
            
        cache = self._load_cache()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Reset cache if it's a new day
        if cache.get('date') != today:
            cache = {'date': today, 'games': {}}
            
        games = cache['games']
        if game_id not in games:
            games[game_id] = {}
            
        team_data = games[game_id].get(team_name, {})
        
        # If no opening odds exist yet, log them now
        if 'opening_odds' not in team_data:
            team_data['opening_odds'] = current_odds
            
        team_data['current_odds'] = current_odds
        games[game_id][team_name] = team_data
        
        self._save_cache(cache)
        
        # Calculate Delta (Implied Probability Shift)
        # e.g., opening = 2.0 (50%), current = 1.8 (55%) -> Delta = +5% steam
        open_prob = 1.0 / team_data['opening_odds']
        curr_prob = 1.0 / current_odds
        
        return curr_prob - open_prob

    def detect_steam(self, game_id, team_name, current_odds, public_bet_pct=0.50):
        """
        Detects sharp 'Reverse Line Movement' (Steam).
        A Steam event occurs when the line moves heavily in favor of a team 
        (probability increases) despite the public betting the other way.
        """
        delta_prob = self.log_odds(game_id, team_name, current_odds)
        
        is_steam = False
        # If probability moved up by >3% but public is NOT heavily backing them
        if delta_prob > 0.03 and public_bet_pct < 0.55:
            is_steam = True
            
        return {
            'delta_prob': delta_prob,
            'is_steam': is_steam
        }
