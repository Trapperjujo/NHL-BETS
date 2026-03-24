import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import time

def optimize_xgboost():
    print("Loading historical training data (~6,700 games)...")
    try:
        df = pd.read_csv("historical_training_data.csv")
    except FileNotFoundError:
        print("ERROR: historical_training_data.csv not found!")
        return

    # Engineer the missing basic metrics for historic data alignment
    df['home_streak'] = 0 
    df['away_streak'] = 0
    df['home_pp_pct'] = 20.0 + (df['home_gf_pg'] - 3.0) * 5.0
    df['away_pp_pct'] = 20.0 + (df['away_gf_pg'] - 3.0) * 5.0
    df['home_pk_pct'] = 80.0 - (df['home_ga_pg'] - 3.0) * 5.0
    df['away_pk_pct'] = 80.0 - (df['away_ga_pg'] - 3.0) * 5.0

    X = df[[
        'home_win_pct', 'away_win_pct', 'home_gf_pg', 'away_gf_pg',
        'home_ga_pg', 'away_ga_pg', 'home_l10_win_pct', 'away_l10_win_pct',
        'home_streak', 'away_streak', 'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct', 'home_shots_diff', 'away_shots_diff',
        'home_xg_for_pg', 'away_xg_for_pg', 'home_xg_against_pg', 'away_xg_against_pg',
        'home_sv_pct', 'away_sv_pct', 'home_is_b2b', 'away_is_b2b',
        'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
        'home_hd_shots_for', 'away_hd_shots_for', 'home_hd_shots_against', 'away_hd_shots_against',
        'home_hd_xg_for', 'away_hd_xg_for', 'home_hd_xg_against', 'away_hd_xg_against',
        'home_sva_xg_for', 'away_sva_xg_for', 'home_sva_xg_against', 'away_sva_xg_against',
        'home_pen_drawn', 'away_pen_drawn', 'home_pen_taken', 'away_pen_taken'
    ]]
    y = df['home_win']

    # Define the parameter grid to test
    # This combination represents dozens of simulated seasons
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)

    # TimeSeriesSplit is crucial for sports so we don't leak future games into past tests
    print("Initializing Grid Search over 500+ parameter combinations...")
    print("This requires heavy computing power and may take several minutes.")
    
    cv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        scoring='neg_log_loss', 
        cv=cv, 
        verbose=1, 
        n_jobs=-1
    )

    start_time = time.time()
    grid_search.fit(X, y)
    end_time = time.time()

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE!")
    print(f"Time Taken: {(end_time - start_time) / 60:.2f} minutes")
    print("="*50)
    print(f"Best Parameters Found:\n{grid_search.best_params_}")
    
    best_logloss = -grid_search.best_score_
    print(f"\nBest Cross-Validated Log-Loss: {best_logloss:.4f}")
    
    # Calculate accuracy with best params
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Historical Win/Loss Accuracy (of Best Model): {acc*100:.2f}%\n")
    print("ACTION REQUIRED: Update predictor.py with these new parameters!")

if __name__ == "__main__":
    optimize_xgboost()
