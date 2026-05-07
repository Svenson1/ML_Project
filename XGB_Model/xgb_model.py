import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Chemins des données
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Data-Sets"


def format_id(df, col_name):
    """Convertir une colonne ID en string au format standardisé"""
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna(subset=[col_name]).copy()
    df['Id'] = df[col_name].astype(int).astype(str)
    return df


# Chargement des données brutes
print("Chargement des datasets...")
train_df = pd.read_csv(DATA_DIR / "results_train.csv")
test_df = pd.read_csv(DATA_DIR / "results_test.csv")

train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)
train_df = train_df.drop(columns=['Gemeinde-Nummer'])
test_df = test_df.drop(columns=['Gemeinde-Nummer'])

# Résultats par commune (données 622)
df_622 = pd.read_excel(
    DATA_DIR / "622.00-result-by-canton-district-and-municipality.xlsx",
    sheet_name="Gemeinden", header=5
)
df_622.columns = df_622.columns.str.strip()
df_622 = format_id(df_622, 'Gemeinde-Nummer')
df_622 = df_622.drop_duplicates(subset=['Id'])
df_622 = df_622.add_suffix('_622').rename(columns={'Id_622': 'Id'})
cols_to_drop = ['Gemeinde-Nombre_622', 'Gemeinde_622', 'Kanton_622']
df_622 = df_622.drop(columns=cols_to_drop, errors='ignore')

# Données démographiques (je-e)
df_jee = pd.read_excel(
    DATA_DIR / "je-e-21.03.01.xlsx",
    sheet_name="Schweiz - Gemeinden", header=5
)
df_jee = format_id(df_jee, 'Number of commune')
df_jee = df_jee.drop_duplicates(subset=['Id'])
df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])
for col in df_jee.columns:
    if col != 'Id':
        df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')

# Données géographiques
df_geo = pd.read_csv(DATA_DIR / "swiss_communes_geodata.csv")
df_geo = format_id(df_geo, 'bfs_id')
df_geo = df_geo.drop_duplicates(subset=['Id'])
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

# Données de revenu
df_income = pd.read_excel(
    DATA_DIR / "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx",
    sheet_name='Gemeinden - Communes'
)
df_income = format_id(df_income, 'gdenr')
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income').rename(columns={'Id_income': 'Id'})
for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')

# Fusion de tous les datasets
print("Fusion des datasets...")
train_merged = train_df.merge(df_622, on='Id', how='left')
train_merged = train_merged.merge(df_jee, on='Id', how='left')
train_merged = train_merged.merge(df_income, on='Id', how='left')
train_merged = train_merged.merge(df_geo, on='Id', how='left')

test_merged = test_df.merge(df_622, on='Id', how='left')
test_merged = test_merged.merge(df_jee, on='Id', how='left')
test_merged = test_merged.merge(df_income, on='Id', how='left')
test_merged = test_merged.merge(df_geo, on='Id', how='left')

print(f"Doublons train : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons test : {test_merged['Id'].duplicated().sum()}")

# Nettoyage des colonnes problématiques
cols_drop = ['PdA/Sol.', 'Settlement and urban area in %', 'gdenr_income', 'ktnr_income']
for col in cols_drop:
    train_merged = train_merged.drop(columns=[col], errors='ignore')
    test_merged = test_merged.drop(columns=[col], errors='ignore')

# Remplir les colonnes de partis manquantes par 0
party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP', 'EVP/CSP', 'Small right-wing parties']
for col in party_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col] = test_merged[col].fillna(0)

# Sauvegarder le canton avant encodage (nécessaire pour le target encoding)
canton_train = train_merged['Kantons-Nummer'].values.copy()
canton_test = test_merged['Kantons-Nummer'].values.copy()

# One-hot encoding du canton
print("One-hot encoding du canton...")
dummies_train = pd.get_dummies(
    train_merged['Kantons-Nummer'], 
    prefix='canton', 
    drop_first=True, 
    dtype=int
)
dummies_test = pd.get_dummies(
    test_merged['Kantons-Nummer'],
    prefix='canton',
    drop_first=True,
    dtype=int
)

# Aligner les colonnes entre train et test
dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

# Ajouter les colonnes one-hot et supprimer le canton original
train_merged = pd.concat([train_merged.drop(columns=['Kantons-Nummer']), dummies_train], axis=1)
test_merged = pd.concat([test_merged.drop(columns=['Kantons-Nummer']), dummies_test], axis=1)


# Feature engineering
def create_features(df):
    """Créer les features engineered pour capturer les non-linéarités"""
    df = df.copy()

    # Population (pour log et interactions)
    pop_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ["resident", "einwohn"]):
            if pd.to_numeric(df[col], errors='coerce').median() > 100:
                pop_col = col
                break

    if pop_col:
        pop = pd.to_numeric(df[pop_col], errors='coerce').fillna(100)
        pop = pop.clip(lower=1)
        df['log_population'] = np.log1p(pop)
        df['sqrt_population'] = np.sqrt(pop)

    # Électeurs
    if 'Stimmberechtigte_622' in df.columns:
        elec = pd.to_numeric(df['Stimmberechtigte_622'], errors='coerce').fillna(100)
        elec = elec.clip(lower=1)
        df['log_electeurs'] = np.log1p(elec)

    # Agriculture (signal dominant, corrélation = -0.623)
    # Créer des transformations non-linéaires car l'effet n'est pas linéaire
    if 'Agricultural area in %' in df.columns:
        agri = pd.to_numeric(df['Agricultural area in %'], errors='coerce').fillna(0)
        df['agri_sq'] = agri ** 2
        df['agri_sqrt'] = np.sqrt(agri.clip(lower=0))

        # Interactions agriculture x démographie
        if '0-19 years' in df.columns:
            young = pd.to_numeric(df['0-19 years'], errors='coerce').fillna(0)
            df['agri_x_young'] = agri * young

        if 'Size of households in persons' in df.columns:
            hh = pd.to_numeric(df['Size of households in persons'], errors='coerce').fillna(0)
            df['agri_x_hh'] = agri * hh

        if '65 years or over' in df.columns:
            senior = pd.to_numeric(df['65 years or over'], errors='coerce').fillna(0)
            df['agri_x_senior'] = agri * senior

        if 'Unproductive area in %' in df.columns:
            unprod = pd.to_numeric(df['Unproductive area in %'], errors='coerce').fillna(0)
            df['agri_x_unprod'] = agri * unprod
            df['ratio_unprod_agri'] = unprod / (agri + 1)

        if pop_col:
            pop = pd.to_numeric(df[pop_col], errors='coerce').fillna(100).clip(lower=1)
            df['agri_x_logpop'] = agri * np.log1p(pop)

    # Ratio emploi agricole
    if 'Agricultural employment' in df.columns and 'Total employment' in df.columns:
        agri_emp = pd.to_numeric(df['Agricultural employment'], errors='coerce').fillna(0)
        tot_emp = pd.to_numeric(df['Total employment'], errors='coerce').fillna(1)
        df['ratio_agri_emploi'] = agri_emp / (tot_emp + 1)

    # Partis de droite x agriculture
    droite_cols = [c for c in ['SVP', 'FDP/PLR 2)', 'CVP', 'Small right-wing parties'] 
                   if c in df.columns]
    if droite_cols and 'Agricultural area in %' in df.columns:
        df['droite_tot'] = df[droite_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        agri = pd.to_numeric(df['Agricultural area in %'], errors='coerce').fillna(0)
        df['droite_x_agri'] = df['droite_tot'] * agri

        if '65 years or over' in df.columns:
            senior = pd.to_numeric(df['65 years or over'], errors='coerce').fillna(0)
            df['droite_x_senior'] = df['droite_tot'] * senior

    # Variables régionales (Suisse romande vs alémanique)
    if 'lon' in df.columns and 'lat' in df.columns:
        lon = pd.to_numeric(df['lon'], errors='coerce')
        df['is_romand'] = (lon < 7.5).astype(float)

        if 'Agricultural area in %' in df.columns:
            agri = pd.to_numeric(df['Agricultural area in %'], errors='coerce').fillna(0)
            df['romand_x_agri'] = df['is_romand'] * agri

    # Revenus en log (distributions asymétriques)
    for rev_col in ['mean_reinka_income', 'median_reinka_income']:
        if rev_col in df.columns:
            val = pd.to_numeric(df[rev_col], errors='coerce').clip(lower=0)
            df[f'log_{rev_col}'] = np.log1p(val)

    return df


print("Feature engineering...")
train_merged = create_features(train_merged)
test_merged = create_features(test_merged)


# Extraction de la cible et sélection des features
y_train = train_merged['Ja in Prozent'].values

# Colonnes qui contiennent des informations du vote (leakage)
leakage = ['eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
           'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent']

# Sélectionner uniquement les features numériques (pas les strings)
X_train_raw = train_merged.select_dtypes(include=[np.number])
X_train_raw = X_train_raw.drop(columns=[c for c in leakage if c in X_train_raw.columns])
X_test_raw = test_merged[X_train_raw.columns]

# Supprimer les colonnes avec plus de 60% de valeurs manquantes
print("Nettoyage des features...")
missing_ratio = X_train_raw.isna().mean()
cols_to_keep = missing_ratio[missing_ratio <= 0.6].index.tolist()
X_train_raw = X_train_raw[cols_to_keep]
X_test_raw = X_test_raw[cols_to_keep]

# Sélection des features par corrélation ou p-valeur
cols_selected = []
for col in X_train_raw.columns:
    vals = X_train_raw[col]
    valid_idx = vals.notna()
    
    if valid_idx.sum() < 20:
        continue
    
    try:
        r, p = pearsonr(vals[valid_idx], y_train[valid_idx])
        # Garder si corrélation suffisante OU p-valeur faible
        if abs(r) >= 0.02 or p < 0.10:
            cols_selected.append(col)
    except:
        # Si erreur (ex: variance 0), ajouter quand même
        cols_selected.append(col)

X_train_raw = X_train_raw[cols_selected]
X_test_raw = X_test_raw[cols_selected]

print(f"Features conservées : {X_train_raw.shape[1]}")
print(f"Valeurs manquantes : {X_train_raw.isna().sum().sum()}")

# Imputation des valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    columns=X_train_raw.columns
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test_raw),
    columns=X_train_raw.columns
)


# Détection des outliers (IQR)
Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1

is_outlier = (y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)
sample_weights = np.where(is_outlier, 0.6, 1.0)

print(f"Outliers détectés : {is_outlier.sum()} communes")


# Entraînement XGBoost avec target encoding in-fold
print("\nEntraînement XGBoost...")

SMOOTHING = 15
GLOBAL_MEAN = y_train.mean()
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.005,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'colsample_bylevel': 0.75,
    'colsample_bynode': 0.75,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'min_child_weight': 8,
    'gamma': 0.0,
    'tree_method': 'hist',
    'n_jobs': -1,
    'n_estimators': 10000,
}

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test_merged))
rmse_scores = []

# Target encoding sur tout le train pour le test
te_map_full = {}
for canton in np.unique(canton_train):
    mask = canton_train == canton
    n_samples = mask.sum()
    mean_val = y_train[mask].mean()
    te_map_full[canton] = (mean_val * n_samples + GLOBAL_MEAN * SMOOTHING) / (n_samples + SMOOTHING)

te_test = np.array([te_map_full.get(c, GLOBAL_MEAN) for c in canton_test])

# Validation croisée
for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train_imp)):
    print(f"Fold {fold_idx + 1}/{N_FOLDS}")
    
    # Target encoding calculé UNIQUEMENT sur le fold d'entraînement
    canton_tr = canton_train[tr_idx]
    y_tr = y_train[tr_idx]
    canton_val = canton_train[val_idx]

    te_map_fold = {}
    for canton in np.unique(canton_tr):
        mask = canton_tr == canton
        n_samples = mask.sum()
        mean_val = y_tr[mask].mean()
        te_map_fold[canton] = (mean_val * n_samples + GLOBAL_MEAN * SMOOTHING) / (n_samples + SMOOTHING)

    te_tr = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_tr])
    te_val = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_val])

    # Ajouter le target encoding aux features
    X_fold_tr = X_train_imp.iloc[tr_idx].copy()
    X_fold_val = X_train_imp.iloc[val_idx].copy()
    X_fold_test = X_test_imp.copy()

    X_fold_tr['canton_te'] = te_tr
    X_fold_val['canton_te'] = te_val
    X_fold_test['canton_te'] = te_test

    y_fold_tr = y_train[tr_idx]
    y_fold_val = y_train[val_idx]
    w_fold_tr = sample_weights[tr_idx]

    # Moyennage sur 5 seeds
    pred_val_fold = np.zeros(len(val_idx))
    pred_test_fold = np.zeros(len(test_merged))

    for seed in SEEDS:
        params_seed = params.copy()
        params_seed['random_state'] = seed

        model = xgb.XGBRegressor(params_seed, early_stopping_rounds=75, eval_metric='rmse')
        model.fit(
            X_fold_tr, y_fold_tr,
            sample_weight=w_fold_tr,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        pred_val = np.clip(model.predict(X_fold_val), 0, 100)
        pred_test = np.clip(model.predict(X_fold_test), 0, 100)

        pred_val_fold += pred_val / len(SEEDS)
        pred_test_fold += pred_test / len(SEEDS)

    oof_preds[val_idx] = pred_val_fold
    test_preds += pred_test_fold / N_FOLDS

    rmse = np.sqrt(mean_squared_error(y_fold_val, pred_val_fold))
    rmse_scores.append(rmse)
    print(f"  RMSE = {rmse:.4f}")

rmse_oof = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\nRMSE OOF final : {rmse_oof:.4f}")

bias = float(np.mean(oof_preds - y_train))
print(f"Biais : {bias:+.4f}")

final_predictions = np.clip(test_preds, 0, 100)


# Graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot prédictions vs réel
axes[0].scatter(y_train, oof_preds, alpha=0.35, s=10, color='steelblue')
axes[0].scatter(y_train[is_outlier], oof_preds[is_outlier],
                alpha=0.9, s=35, color='red', label='Outliers', zorder=5)
axes[0].plot([0, 100], [0, 100], 'r--', lw=1.5)
axes[0].set_xlabel("% OUI réel")
axes[0].set_ylabel("% OUI prédit")
axes[0].set_title(f"Prédictions OOF - RMSE = {rmse_oof:.4f}")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Distribution des erreurs
errors = y_train - oof_preds
axes[1].hist(errors, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axes[1].axvline(0, color='red', ls='--', lw=2)
axes[1].set_xlabel("Erreur résiduelle")
axes[1].set_ylabel("Nombre de communes")
axes[1].set_title(f"Distribution des erreurs (biais={bias:+.3f})")
axes[1].grid(True, alpha=0.3)

# RMSE par fold
axes[2].bar(range(1, N_FOLDS + 1), rmse_scores, color='steelblue', alpha=0.85, edgecolor='black')
axes[2].axhline(rmse_oof, color='red', ls='--', lw=2, label=f'Moyenne = {rmse_oof:.4f}')
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("RMSE")
axes[2].set_title("Performance par fold")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgb_results.png', dpi=150)
plt.close()

# Feature importance
X_full = X_train_imp.copy()
X_full['canton_te'] = np.array([te_map_full.get(c, GLOBAL_MEAN) for c in canton_train])

model_full = xgb.XGBRegressor(**params)
model_full.fit(X_full, y_train, sample_weight=sample_weights, verbose=False)

importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top25 = importance_df.head(25)
ax.barh(range(len(top25)), top25['importance'].values, color='steelblue', alpha=0.85)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25['feature'].values)
ax.set_xlabel('Importance (gain)')
ax.set_title('XGBoost - Top 25 Features')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=150)
plt.close()

print("\nTop 10 features :")
print(importance_df.head(10).to_string(index=False))

# Soumission
submission = pd.DataFrame({
    'Id': test_merged['Id'],
    'Predicted': np.clip(final_predictions, 0, 100)
})
submission.to_csv('submission_xgboost.csv', index=False)
print("\nSubmission sauvegardée : submission_xgboost.csv")
print(submission.head())
