import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ========================================
# CHARGEMENT ET FUSION DES DONNÉES
# ========================================

print("Chargement des datasets...")

# Déterminer le chemin correct (depuis XGB_model/)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "Data_Sets")

# Train et test
train_df = pd.read_csv(os.path.join(data_dir, "results_train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "results_test.csv"))
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)

# Données 622
df_622 = pd.read_excel(os.path.join(data_dir, "622.00-result-by-canton-district-and-municipality.xlsx"),
                        sheet_name="Gemeinden", header=5)
df_622.columns = df_622.columns.str.strip()
df_622['Id'] = df_622['Gemeinde-Nummer'].astype(str)
df_622 = df_622.drop_duplicates(subset=['Id'])
df_622 = df_622.add_suffix('_622').rename(columns={'Id_622': 'Id'})

# Données démographiques
df_jee = pd.read_excel(os.path.join(data_dir, "je-e-21.03.01.xlsx"),
                        sheet_name="Schweiz - Gemeinden", header=5)
df_jee['Id'] = df_jee['Number of commune'].astype(str)
df_jee = df_jee.drop_duplicates(subset=['Id'])

# Données géographiques
df_geo = pd.read_csv(os.path.join(data_dir, "swiss_communes_geodata.csv"))
df_geo['Id'] = df_geo['bfs_id'].astype(str)
df_geo = df_geo.drop_duplicates(subset=['Id'])

# Données de revenu
df_income = pd.read_excel(os.path.join(data_dir, "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"),
                           sheet_name='Gemeinden - Communes')
df_income['Id'] = df_income['gdenr'].astype(str)
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.add_suffix('_income').rename(columns={'Id_income': 'Id'})

# Fusion LEFT JOIN
train_merged = train_df.merge(df_622, on='Id', how='left')
train_merged = train_merged.merge(df_jee, on='Id', how='left')
train_merged = train_merged.merge(df_income, on='Id', how='left')
train_merged = train_merged.merge(df_geo, on='Id', how='left')

test_merged = test_df.merge(df_622, on='Id', how='left')
test_merged = test_merged.merge(df_jee, on='Id', how='left')
test_merged = test_merged.merge(df_income, on='Id', how='left')
test_merged = test_merged.merge(df_geo, on='Id', how='left')

print(f"Train : {train_merged.shape[0]} communes x {train_merged.shape[1]} features")
print(f"Test : {test_merged.shape[0]} communes x {test_merged.shape[1]} features")

# ========================================
# NETTOYAGE DES DONNÉES
# ========================================

print("Nettoyage des données...")

# Supprimer les colonnes problématiques
cols_to_drop = ['PdA/Sol.', 'Settlement and urban area in %', 'gdenr_income', 'ktnr_income']
for col in cols_to_drop:
    if col in train_merged.columns:
        train_merged = train_merged.drop(columns=[col])
        test_merged = test_merged.drop(columns=[col])

# Remplir les partis politiques manquants par 0
party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP', 'EVP/CSP', 'Small right-wing parties']
for col in party_cols:
    if col in train_merged.columns:
        train_merged[col] = pd.to_numeric(train_merged[col], errors='coerce').fillna(0)
        test_merged[col] = pd.to_numeric(test_merged[col], errors='coerce').fillna(0)

# Sauvegarder le canton (nécessaire pour target encoding)
canton_train = train_merged['Kantons-Nummer'].values.copy()
canton_test = test_merged['Kantons-Nummer'].values.copy()

# One-hot encoding du canton
print("One-hot encoding du canton...")
dummies_train = pd.get_dummies(train_merged['Kantons-Nummer'], prefix='canton', drop_first=True, dtype=int)
dummies_test = pd.get_dummies(test_merged['Kantons-Nummer'], prefix='canton', drop_first=True, dtype=int)
dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

train_merged = pd.concat([train_merged.drop(columns=['Kantons-Nummer']), dummies_train], axis=1)
test_merged = pd.concat([test_merged.drop(columns=['Kantons-Nummer']), dummies_test], axis=1)

# ========================================
# FEATURE ENGINEERING
# ========================================

print("Feature engineering...")

# Transformations non-linéaires de l'agriculture
if 'Agricultural area in %' in train_merged.columns:
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    train_merged['agri_sq'] = agri ** 2
    train_merged['agri_sqrt'] = np.sqrt(agri)
    
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    test_merged['agri_sq'] = agri_test ** 2
    test_merged['agri_sqrt'] = np.sqrt(agri_test)

# Interactions agriculture x jeunes
if 'Agricultural area in %' in train_merged.columns and '0-19 years' in train_merged.columns:
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    young = pd.to_numeric(train_merged['0-19 years'], errors='coerce').fillna(0)
    train_merged['agri_x_young'] = agri * young
    
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    young_test = pd.to_numeric(test_merged['0-19 years'], errors='coerce').fillna(0)
    test_merged['agri_x_young'] = agri_test * young_test

# Interactions agriculture x ménages
if 'Agricultural area in %' in train_merged.columns and 'Size of households in persons' in train_merged.columns:
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    hh = pd.to_numeric(train_merged['Size of households in persons'], errors='coerce').fillna(0)
    train_merged['agri_x_hh'] = agri * hh
    
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    hh_test = pd.to_numeric(test_merged['Size of households in persons'], errors='coerce').fillna(0)
    test_merged['agri_x_hh'] = agri_test * hh_test

# Interactions agriculture x seniors
if 'Agricultural area in %' in train_merged.columns and '65 years or over' in train_merged.columns:
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    senior = pd.to_numeric(train_merged['65 years or over'], errors='coerce').fillna(0)
    train_merged['agri_x_senior'] = agri * senior
    
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    senior_test = pd.to_numeric(test_merged['65 years or over'], errors='coerce').fillna(0)
    test_merged['agri_x_senior'] = agri_test * senior_test

# Interactions agriculture x zones improductives
if 'Agricultural area in %' in train_merged.columns and 'Unproductive area in %' in train_merged.columns:
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    unprod = pd.to_numeric(train_merged['Unproductive area in %'], errors='coerce').fillna(0)
    train_merged['agri_x_unprod'] = agri * unprod
    
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    unprod_test = pd.to_numeric(test_merged['Unproductive area in %'], errors='coerce').fillna(0)
    test_merged['agri_x_unprod'] = agri_test * unprod_test

# Partis de droite x agriculture - FIX: convertir en numeric
droite_cols = ['SVP', 'FDP/PLR 2)', 'CVP', 'Small right-wing parties']
if 'Agricultural area in %' in train_merged.columns:
    # Convertir les colonnes en numeric d'abord
    for col in droite_cols:
        if col in train_merged.columns:
            train_merged[col] = pd.to_numeric(train_merged[col], errors='coerce').fillna(0)
            test_merged[col] = pd.to_numeric(test_merged[col], errors='coerce').fillna(0)
    
    train_merged['droite_tot'] = train_merged[droite_cols].sum(axis=1)
    agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
    train_merged['droite_x_agri'] = train_merged['droite_tot'] * agri
    
    test_merged['droite_tot'] = test_merged[droite_cols].sum(axis=1)
    agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
    test_merged['droite_x_agri'] = test_merged['droite_tot'] * agri_test

# Logarithmes de population
if any(col in train_merged.columns for col in ['Resident population', 'Einwohner']):
    pop_col = [col for col in train_merged.columns if col in ['Resident population', 'Einwohner']][0]
    train_merged['log_population'] = np.log1p(pd.to_numeric(train_merged[pop_col], errors='coerce').fillna(100))
    test_merged['log_population'] = np.log1p(pd.to_numeric(test_merged[pop_col], errors='coerce').fillna(100))

# Logarithmes des revenus
if 'mean_reinka_income' in train_merged.columns:
    train_merged['log_mean_income'] = np.log1p(pd.to_numeric(train_merged['mean_reinka_income'], errors='coerce').fillna(0).clip(lower=0))
    test_merged['log_mean_income'] = np.log1p(pd.to_numeric(test_merged['mean_reinka_income'], errors='coerce').fillna(0).clip(lower=0))

if 'median_reinka_income' in train_merged.columns:
    train_merged['log_median_income'] = np.log1p(pd.to_numeric(train_merged['median_reinka_income'], errors='coerce').fillna(0).clip(lower=0))
    test_merged['log_median_income'] = np.log1p(pd.to_numeric(test_merged['median_reinka_income'], errors='coerce').fillna(0).clip(lower=0))

# Variables régionales
if 'lon' in train_merged.columns:
    train_merged['is_romand'] = (pd.to_numeric(train_merged['lon'], errors='coerce') < 7.5).astype(float)
    test_merged['is_romand'] = (pd.to_numeric(test_merged['lon'], errors='coerce') < 7.5).astype(float)
    
    if 'Agricultural area in %' in train_merged.columns:
        agri = pd.to_numeric(train_merged['Agricultural area in %'], errors='coerce').fillna(0)
        train_merged['romand_x_agri'] = train_merged['is_romand'] * agri
        
        agri_test = pd.to_numeric(test_merged['Agricultural area in %'], errors='coerce').fillna(0)
        test_merged['romand_x_agri'] = test_merged['is_romand'] * agri_test

# ========================================
# SÉLECTION DES FEATURES
# ========================================

print("Sélection des features...")

y_train = pd.to_numeric(train_merged['Ja in Prozent'], errors='coerce').values

# Colonnes de leakage à supprimer
leakage = ['eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
           'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent']

# Sélectionner uniquement les features numériques
X_train = train_merged.select_dtypes(include=[np.number])
X_train = X_train.drop(columns=[c for c in leakage if c in X_train.columns])
X_test = test_merged[X_train.columns]

# Supprimer les colonnes avec plus de 60% de NaN
missing_ratio = X_train.isna().mean()
cols_keep = missing_ratio[missing_ratio <= 0.6].index.tolist()
X_train = X_train[cols_keep]
X_test = X_test[cols_keep]

# Filtrer par corrélation et p-valeur
cols_selected = []
for col in X_train.columns:
    vals = X_train[col].dropna()
    target_vals = y_train[X_train[col].notna()]
    
    if len(vals) < 20:
        continue
    
    try:
        r, p = pearsonr(vals, target_vals)
        # Garder si corrélation forte OU p-valeur faible
        if abs(r) >= 0.02 or p < 0.10:
            cols_selected.append(col)
    except:
        cols_selected.append(col)

X_train = X_train[cols_selected]
X_test = X_test[cols_selected]

print(f"Features conservées : {X_train.shape[1]}")

# Imputation des NaN restants
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

# ========================================
# DÉTECTION DES OUTLIERS
# ========================================

Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1
is_outlier = (y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)
sample_weights = np.where(is_outlier, 0.6, 1.0)

print(f"Outliers détectés : {is_outlier.sum()} communes")

# ========================================
# ENTRAÎNEMENT XGBOOST
# ========================================

print("\nEntraînement XGBoost...")

SMOOTHING = 25
GLOBAL_MEAN = y_train.mean()
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.026,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.70,
    'reg_lambda': 1.1,
    'n_estimators': 3000,
}

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test_merged))
rmse_scores = []

# Target encoding sur tout le train
te_map_full = {}
for canton in np.unique(canton_train):
    mask = canton_train == canton
    mean_val = y_train[mask].mean()
    n_samples = mask.sum()
    te_map_full[canton] = (mean_val * n_samples + GLOBAL_MEAN * SMOOTHING) / (n_samples + SMOOTHING)

te_test = np.array([te_map_full.get(c, GLOBAL_MEAN) for c in canton_test])

# Validation croisée
for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
    print(f"Fold {fold_idx + 1}/{N_FOLDS}")
    
    # Target encoding in-fold
    canton_tr = canton_train[tr_idx]
    y_tr = y_train[tr_idx]
    
    te_map_fold = {}
    for canton in np.unique(canton_tr):
        mask = canton_tr == canton
        mean_val = y_tr[mask].mean()
        n_samples = mask.sum()
        te_map_fold[canton] = (mean_val * n_samples + GLOBAL_MEAN * SMOOTHING) / (n_samples + SMOOTHING)
    
    te_tr = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_tr])
    te_val = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_train[val_idx]])
    
    # Ajouter target encoding
    X_tr = X_train.iloc[tr_idx].copy()
    X_tr['canton_te'] = te_tr
    X_val = X_train.iloc[val_idx].copy()
    X_val['canton_te'] = te_val
    X_te = X_test.copy()
    X_te['canton_te'] = te_test
    
    y_tr_fold = y_train[tr_idx]
    y_val_fold = y_train[val_idx]
    w_tr = sample_weights[tr_idx]
    
    # Moyennage sur 5 seeds
    val_preds = np.zeros(len(val_idx))
    test_preds_fold = np.zeros(len(test_merged))
    
    for seed in SEEDS:
        params_seed = params.copy()
        params_seed['random_state'] = seed
        
        model = xgb.XGBRegressor(**params_seed, early_stopping_rounds=75, eval_metric='rmse')
        model.fit(X_tr, y_tr_fold, sample_weight=w_tr, eval_set=[(X_val, y_val_fold)], verbose=False)
        
        val_preds += np.clip(model.predict(X_val), 0, 100) / len(SEEDS)
        test_preds_fold += np.clip(model.predict(X_te), 0, 100) / len(SEEDS)
    
    oof_preds[val_idx] = val_preds
    test_preds += test_preds_fold / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    rmse_scores.append(rmse)
    print(f"  RMSE = {rmse:.4f}")

rmse_oof = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\nRMSE OOF : {rmse_oof:.4f}")

# ========================================
# GRAPHIQUES
# ========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_train, oof_preds, alpha=0.35, s=10)
axes[0].scatter(y_train[is_outlier], oof_preds[is_outlier], alpha=0.9, s=35, color='red')
axes[0].plot([0, 100], [0, 100], 'r--', lw=1.5)
axes[0].set_xlabel("% OUI réel")
axes[0].set_ylabel("% OUI prédit")
axes[0].set_title(f"XGBoost OOF - RMSE = {rmse_oof:.4f}")

axes[1].hist(y_train - oof_preds, bins=40, edgecolor='black')
axes[1].set_xlabel("Erreur résiduelle")
axes[1].set_title("Distribution des erreurs")

axes[2].bar(range(1, N_FOLDS + 1), rmse_scores)
axes[2].axhline(rmse_oof, color='red', ls='--', lw=2)
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("RMSE")
axes[2].set_title("Performance par fold")

plt.tight_layout()
plt.savefig('xgb_results.png', dpi=150)
plt.close()

# ========================================
# SOUMISSION
# ========================================

submission = pd.DataFrame({
    'Id': test_merged['Id'],
    'Predicted': np.clip(test_preds, 0, 100)
})
submission.to_csv('submission_xgboost.csv', index=False)
print(f"\nSubmission sauvegardée : submission_xgboost.csv")
print(submission.head())
