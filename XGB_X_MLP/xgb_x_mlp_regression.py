import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Data_Sets.dataset_utils import import_and_merge_dataset
from Multi_Layer_Perceptron_Regression.mlp_utils import change_train_dataset, create_target

np.random.seed(42)

# ________________________________________________________
# Chargement et merge des Dataset
# ________________________________________________________

train_merged, test_merged = import_and_merge_dataset()

# ________________________________________________________
# Nettoyage
# ________________________________________________________

train_merged, test_merged = change_train_dataset(train_merged, test_merged)

# ________________________________________________________
# Target + One-Hot Kanton
# ________________________________________________________

y_train, train_merged, test_merged = create_target(train_merged, test_merged)

# ________________________________________________________
# Features
# ________________________________________________________

leakage_columns = [
    'eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
    'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent'
]
X_train_raw = train_merged.select_dtypes(include=[np.number]).drop(
    columns=[c for c in leakage_columns if c in train_merged.columns]
)
X_test_raw = test_merged[X_train_raw.columns]

print(f"Features : {X_train_raw.shape[1]} | Train : {X_train_raw.shape[0]} | Test : {X_test_raw.shape[0]}")


# ________________________________________________________
# MLP + Bagging
# ________________________________________________________

print("ÉTAPE 1 : Entraînement MLP + Bagging")

pipeline_mlp = Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('selector', SelectKBest(score_func=lambda X, y: mutual_info_regression(X, y, random_state=42), k='all')),
    ('mlp',      MLPRegressor(
        hidden_layer_sizes=(256,),
        activation='tanh',
        solver='adam',
        alpha=5,
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=4000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )),
])

bagged_mlp = BaggingRegressor(
    estimator=pipeline_mlp,
    n_estimators=25,
    max_samples=0.85,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
    verbose=1,
)

# CV pour obtenir le RMSE du MLP baggé
print("Cross-validation MLP baggé (13 folds)...")
cv_mlp = cross_val_score(
    bagged_mlp,
    X_train_raw, y_train,
    cv=13,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,
)
rmse_mlp = -cv_mlp.mean()  # ← FIX : .mean() et non .best_score_
print(f"RMSE MLP baggé : {rmse_mlp:.3f} ± {(-cv_mlp).std():.3f}")

# Entraînement final du MLP sur tout le train
print("Entraînement final MLP baggé...")
bagged_mlp.fit(X_train_raw, y_train)
print("MLP baggé terminé.")


# ________________________________________________________
# Grid Search XGBoost
# ________________________________________________________

print("ÉTAPE 2 : Recherche des meilleurs paramètres XGBoost")

pipeline_xgb = Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('selector', SelectKBest(score_func=mutual_info_regression)),
    ('xgb',      XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        verbosity=0,
    )),
])

param_grid_xgb = {
    'selector__k': [50, 70, 'all'],
    'xgb__n_estimators': [300, 500, 700],
    'xgb__max_depth': [3, 4, 5, 6],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 0.9],
    'xgb__colsample_bytree': [0.6, 0.7, 0.8],
    'xgb__reg_alpha':  [0, 0.1, 0.5, 1.0],
    'xgb__reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'xgb__min_child_weight': [1, 3, 5],
}

gs_xgb = RandomizedSearchCV(
    pipeline_xgb,
    param_distributions=param_grid_xgb,
    n_iter=80,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
gs_xgb.fit(X_train_raw, y_train)

print(f"Meilleurs paramètres XGB : {gs_xgb.best_params_}")
print(f"Meilleur RMSE XGB        : {-gs_xgb.best_score_:.3f}")
best_xgb = gs_xgb.best_estimator_
rmse_xgb = -gs_xgb.best_score_


# ________________________________________________________
# ÉTAPE 3 — Calcul des poids automatiques
# ________________________________________________________

print("ÉTAPE 3 : Calcul des poids de l'ensemble")

w_mlp = round(1 / rmse_mlp, 4)
w_xgb = round(1 / rmse_xgb, 4)
total = w_mlp + w_xgb
w_mlp_norm = round(w_mlp / total * 10, 2)
w_xgb_norm = round(w_xgb / total * 10, 2)

print(f"RMSE MLP baggé : {rmse_mlp:.3f}  →  poids MLP : {w_mlp_norm:.2f}")
print(f"RMSE XGB       : {rmse_xgb:.3f}  →  poids XGB : {w_xgb_norm:.2f}")


# ________________________________________________________
# ÉTAPE 4 — Ensemble MLP + XGBoost
# ________________________________________________________

print("ÉTAPE 4 : Évaluation de l'ensemble")

ensemble = VotingRegressor(
    estimators=[
        ('mlp', bagged_mlp),
        ('xgb', best_xgb),
    ],
    weights=[w_mlp_norm, w_xgb_norm],
    n_jobs=1,
)

cv_ensemble = cross_val_score(
    ensemble, X_train_raw, y_train,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,
)
rmse_ensemble = -cv_ensemble.mean()
print(f"RMSE Ensemble : {rmse_ensemble:.3f} ± {cv_ensemble.std():.3f}")

print("\n--- Comparaison ---")
print(f"MLP baggé : {rmse_mlp:.3f}")
print(f"XGB       : {rmse_xgb:.3f}")
print(f"Ensemble  : {rmse_ensemble:.3f}  ← {'Meilleur !' if rmse_ensemble < min(rmse_mlp, rmse_xgb) else 'Pas mieux que le meilleur individuel'}")


# ________________________________________________________
# ÉTAPE 5 — Entraînement final + Soumission
# ________________________________________________________

print("ÉTAPE 5 : Entraînement final et soumission")

best_rmse  = min(rmse_mlp, rmse_xgb, rmse_ensemble)
best_model = {rmse_mlp: bagged_mlp, rmse_xgb: best_xgb, rmse_ensemble: ensemble}[best_rmse]
best_label = {rmse_mlp: 'mlp_bagged', rmse_xgb: 'xgboost', rmse_ensemble: 'ensemble'}[best_rmse]

print(f"Modèle retenu : {best_label} (RMSE = {best_rmse:.3f})")
best_model.fit(X_train_raw, y_train)

predictions = np.clip(best_model.predict(X_test_raw), 0, 100)
submission = pd.DataFrame({
    'Id':        test_merged['Id'],
    'Predicted': predictions
})
submission.to_csv(f'submission_{best_label}_final.csv', index=False)
print(f"Submission sauvegardée : submission_{best_label}_final.csv")
print(submission.head())