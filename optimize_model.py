import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import time

print("Loading historical training data (~6,700 games)...")
try:
    df = pd.read_csv("historical_training_data.csv")
    
    # Engineer the missing basic metrics for historic data alignment
    df['home_streak'] = 0 
    df['away_streak'] = 0
    df['home_pp_pct'] = 20.0 + (df['home_gf_pg'] - 3.0) * 5.0
    df['away_pp_pct'] = 20.0 + (df['away_gf_pg'] - 3.0) * 5.0
    df['home_pk_pct'] = 80.0 - (df['home_ga_pg'] - 3.0) * 5.0
    df['away_pk_pct'] = 80.0 - (df['away_ga_pg'] - 3.0) * 5.0
    
    # Phase 5: HDSV% Backfill
    df['home_hdsv_pct'] = df['home_sv_pct'] - 0.080
    df['away_hdsv_pct'] = df['away_sv_pct'] - 0.080
    df['home_flight_fatigue'] = 1.0
    df['away_flight_fatigue'] = 1.0

    X = df[[
        'home_win_pct', 'away_win_pct', 'home_gf_pg', 'away_gf_pg',
        'home_ga_pg', 'away_ga_pg', 'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct', 'home_shots_diff', 'away_shots_diff',
        'home_xg_for_pg', 'away_xg_for_pg', 'home_xg_against_pg', 'away_xg_against_pg',
        'home_sv_pct', 'away_sv_pct', 'home_hdsv_pct', 'away_hdsv_pct',
        'home_cf_pct', 'away_cf_pct', 'home_ff_pct', 'away_ff_pct',
        'home_hd_shots_for', 'away_hd_shots_for', 'home_hd_shots_against', 'away_hd_shots_against',
        'home_hd_xg_for', 'away_hd_xg_for', 'home_hd_xg_against', 'away_hd_xg_against',
        'home_sva_xg_for', 'away_sva_xg_for', 'home_sva_xg_against', 'away_sva_xg_against',
        'home_pen_drawn', 'away_pen_drawn', 'home_pen_taken', 'away_pen_taken',
        'home_elo', 'away_elo', 'home_flight_fatigue', 'away_flight_fatigue'
    ]]
    y = df['home_win']
except FileNotFoundError:
    print("ERROR: historical_training_data.csv not found!")
    X, y = None, None

def objective(trial):
    # Suggest hyperparameters using Optuna's intelligent search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    cv = TimeSeriesSplit(n_splits=3)
    loglosses = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)
        ll = log_loss(y_val, preds)
        loglosses.append(ll)
        
    return sum(loglosses) / len(loglosses)

def optimize_xgboost():
    if X is None:
        return
        
    print("Initializing Optuna Hyperparameter Automation...")
    print("This will mathematically hunt for the absolute BEST architecture over multiple trials.")
    
    # Run 50 trials for an intensive live session search
    study = optuna.create_study(direction='minimize')
    start_time = time.time()
    study.optimize(objective, n_trials=50)
    end_time = time.time()

    print("\n" + "="*50)
    print("OPTUNA OPTIMIZATION COMPLETE!")
    print(f"Time Taken: {(end_time - start_time) / 60:.2f} minutes")
    print("="*50)
    print(f"Best Multi-Season Log-Loss: {study.best_value:.4f}")
    print(f"\nAbsolute Best Parameters Found:\n{study.best_params}")
    
    print("\nACTION REQUIRED: Copy these parameters into the XGBClassifier inside predictor.py!")

if __name__ == "__main__":
    optimize_xgboost()
