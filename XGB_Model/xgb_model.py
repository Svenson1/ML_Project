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

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "Data-Sets"


# ________________________________________________________
# Chargement des Dataset
# ________________________________________________________

def clean_and_format_id(df, column_name):
    """
    :param df: The initial DataFrame
    :param column_name: The name of the column
    :return: the cleaned DataFrame with the format ID
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name]).copy()
    df['Id'] = df[column_name].astype(int).astype(str)
    return df


train_df = pd.read_csv(DATA_DIR / "results_train.csv")
test_df  = pd.read_csv(DATA_DIR / "results_test.csv")
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id']  = test_df['Gemeinde-Nummer'].astype(str)
train_df = train_df.drop(columns=['Gemeinde-Nummer'])
test_df  = test_df.drop(columns=['Gemeinde-Nummer'])

df_622 = pd.read_excel(DATA_DIR / "622.00-result-by-canton-district-and-municipality.xlsx",
                        sheet_name="Gemeinden", header=5)
df_622.columns = df_622.columns.str.strip()
df_622 = clean_and_format_id(df_622, 'Gemeinde-Nummer')
df_622 = df_622.drop_duplicates(subset=['Id'])
df_622 = df_622.add_suffix('_622').rename(columns={'Id_622': 'Id'})
df_622 = df_622.drop(columns=['Gemeinde-Nombre_622', 'Gemeinde_622', 'Kanton_622'], errors='ignore')

df_jee = pd.read_excel(DATA_DIR / "je-e-21.03.01.xlsx",
                        sheet_name="Schweiz - Gemeinden", header=5)
df_jee = clean_and_format_id(df_jee, 'Number of commune')
df_jee = df_jee.drop_duplicates(subset=['Id'])
df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])
for col in df_jee.columns:
    if col != 'Id':
        df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')

df_geo = pd.read_csv(DATA_DIR / "swiss_communes_geodata.csv")
df_geo = clean_and_format_id(df_geo, 'bfs_id')
df_geo = df_geo.drop_duplicates(subset=['Id'])
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

df_income = pd.read_excel(DATA_DIR / "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx",
                           sheet_name='Gemeinden - Communes')
df_income = clean_and_format_id(df_income, 'gdenr')
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income').rename(columns={'Id_income': 'Id'})
for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')


# ________________________________________________________
# Merge Datasets
# ________________________________________________________

train_merged = (train_df.merge(df_622,    on='Id', how='left')
                        .merge(df_jee,    on='Id', how='left')
                        .merge(df_income, on='Id', how='left')
                        .merge(df_geo,    on='Id', how='left'))

test_merged  = (test_df.merge(df_622,    on='Id', how='left')
                       .merge(df_jee,    on='Id', how='left')
                       .merge(df_income, on='Id', how='left')
                       .merge(df_geo,    on='Id', how='left'))

print(f"Doublons dans train_merged : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons dans test_merged  : {test_merged['Id'].duplicated().sum()}")


# ________________________________________________________
# Nettoyage
# ________________________________________________________

for col in ['PdA/Sol.', 'Settlement and urban area in %', 'gdenr_income', 'ktnr_income']:
    train_merged = train_merged.drop(columns=[col], errors='ignore')
    test_merged  = test_merged.drop(columns=[col],  errors='ignore')

party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP', 'EVP/CSP', 'Small right-wing parties']
for col in party_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col]  = test_merged[col].fillna(0)

# Sauvegarder le canton avant encodage (nécessaire pour le target encoding in-fold)
canton_train = train_merged['Kantons-Nummer'].values.copy()
canton_test  = test_merged['Kantons-Nummer'].values.copy()

# One-hot canton
dummies_train = pd.get_dummies(train_merged['Kantons-Nummer'], prefix='canton', drop_first=True, dtype=int)
dummies_test  = pd.get_dummies(test_merged['Kantons-Nummer'],  prefix='canton', drop_first=True, dtype=int)
dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

train_merged = pd.concat([train_merged.drop(columns=['Kantons-Nummer']), dummies_train], axis=1)
test_merged  = pd.concat([test_merged.drop(columns=['Kantons-Nummer']),  dummies_test],  axis=1)


# ________________________________________________________
# Feature Engineering
# ________________________________________________________

def add_features(df):
    df = df.copy()

    # Population
    pop_col = next((c for c in df.columns if any(kw in c.lower() for kw in ["resident", "einwohn"])
                    and pd.to_numeric(df[c], errors='coerce').median() > 100), None)
    if pop_col:
        pop = pd.to_numeric(df[pop_col], errors='coerce').fillna(100).clip(lower=1)
        df["log_population"]  = np.log1p(pop)
        df["sqrt_population"] = np.sqrt(pop)

    if "Stimmberechtigte_622" in df.columns:
        elec = pd.to_numeric(df["Stimmberechtigte_622"], errors='coerce').fillna(100).clip(lower=1)
        df["log_electeurs"] = np.log1p(elec)

    # Agriculture (r = -0.623, signal dominant)
    if "Agricultural area in %" in df.columns:
        agri = pd.to_numeric(df["Agricultural area in %"], errors='coerce').fillna(0)
        df["agri_sq"]   = agri ** 2
        df["agri_sqrt"] = np.sqrt(agri.clip(lower=0))

        if "0-19 years" in df.columns:
            df["agri_x_young"] = agri * pd.to_numeric(df["0-19 years"], errors='coerce').fillna(0)
        if "Size of households in persons" in df.columns:
            df["agri_x_hh"] = agri * pd.to_numeric(df["Size of households in persons"], errors='coerce').fillna(0)
        if "65 years or over" in df.columns:
            df["agri_x_senior"] = agri * pd.to_numeric(df["65 years or over"], errors='coerce').fillna(0)
        if "Unproductive area in %" in df.columns:
            unprod = pd.to_numeric(df["Unproductive area in %"], errors='coerce').fillna(0)
            df["agri_x_unprod"]     = agri * unprod
            df["ratio_unprod_agri"] = unprod / (agri + 1)
        if pop_col:
            pop = pd.to_numeric(df[pop_col], errors='coerce').fillna(100).clip(lower=1)
            df["agri_x_logpop"] = agri * np.log1p(pop)

    if "Agricultural employment" in df.columns and "Total employment" in df.columns:
        df["ratio_agri_emploi"] = (
            pd.to_numeric(df["Agricultural employment"], errors='coerce').fillna(0)
            / (pd.to_numeric(df["Total employment"], errors='coerce').fillna(1) + 1)
        )

    # Politique x agriculture
    droite_cols = [c for c in ["SVP", "FDP/PLR 2)", "CVP", "Small right-wing parties"] if c in df.columns]
    if droite_cols:
        df["droite_tot"] = df[droite_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        if "Agricultural area in %" in df.columns:
            agri = pd.to_numeric(df["Agricultural area in %"], errors='coerce').fillna(0)
            df["droite_x_agri"] = df["droite_tot"] * agri
            if "65 years or over" in df.columns:
                df["droite_x_senior"] = df["droite_tot"] * pd.to_numeric(df["65 years or over"], errors='coerce').fillna(0)

    # Régions linguistiques via GPS
    # DIAGNOSTIC AXE 2 : is_tessin r=0.36 mais redondant avec canton_te (canton 21)
    # DIAGNOSTIC AXE 4 : SANS is_tessin → RMSE 5.997 vs AVEC → 6.016 → on retire is_tessin
    # On garde is_romand (r=-0.245, signal indépendant du canton_te) et romand_x_agri
    if "lon" in df.columns and "lat" in df.columns:
        lon = pd.to_numeric(df["lon"], errors='coerce')
        lat = pd.to_numeric(df["lat"], errors='coerce')
        df["is_romand"] = (lon < 7.5).astype(float)
        # is_tessin SUPPRIME : redondant avec canton_te, cause overfitting sur les 141 communes tessinoises
        if "Agricultural area in %" in df.columns:
            agri = pd.to_numeric(df["Agricultural area in %"], errors='coerce').fillna(0)
            df["romand_x_agri"] = df["is_romand"] * agri

    # Revenus (log pour distributions asymétriques)
    for rev_col in ["mean_reinka_income", "median_reinka_income"]:
        if rev_col in df.columns:
            val = pd.to_numeric(df[rev_col], errors='coerce').clip(lower=0)
            df[f"log_{rev_col}"] = np.log1p(val)

    return df

train_merged = add_features(train_merged)
test_merged  = add_features(test_merged)


# ________________________________________________________
# Sélection des features
# ________________________________________________________

y_train = train_merged['Ja in Prozent'].values

leakage_columns = ['eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
                   'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent']

X_train_raw = train_merged.select_dtypes(include=[np.number]).drop(
    columns=[c for c in leakage_columns if c in train_merged.columns])
X_test_raw  = test_merged[X_train_raw.columns]

missing_ratio = X_train_raw.isna().mean()
X_train_raw   = X_train_raw.drop(columns=missing_ratio[missing_ratio > 0.6].index.tolist())
X_test_raw    = X_test_raw[X_train_raw.columns]

cols_a_garder = []
for col in X_train_raw.columns:
    vals  = X_train_raw[col]
    valid = vals.notna()
    if valid.sum() < 20:
        continue
    try:
        r, p = pearsonr(vals[valid], y_train[valid])
        if abs(r) >= 0.02 or p < 0.10:
            cols_a_garder.append(col)
    except Exception:
        cols_a_garder.append(col)

X_train_raw = X_train_raw[cols_a_garder]
X_test_raw  = X_test_raw[cols_a_garder]

print(f"Features : {X_train_raw.shape[1]} colonnes")
print(f"NaN dans X_train : {X_train_raw.isna().sum().sum()}")

imputer     = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw),      columns=X_train_raw.columns)


# ________________________________________________________
# Gestion des outliers
# ________________________________________________________

Q1, Q3 = np.percentile(y_train, [25, 75])
IQR    = Q3 - Q1
is_outlier     = (y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)
sample_weights = np.where(is_outlier, 0.6, 1.0)
print(f"Outliers IQR : {is_outlier.sum()} communes (poids 0.6)")


# ________________________________________________________
# XGBoost - Target Encoding IN-FOLD + 5 Seeds
# ________________________________________________________
# DIAGNOSTIC AXE 3 : 5-fold std=0.089 vs 7-fold std=0.297 → 5-fold plus stable
# DIAGNOSTIC AXE 4 : is_tessin retiré → gain de 0.019 RMSE
# Target encoding in-fold : évite la fuite d'information du fold val vers canton_te

print("\n_____ XGBoost - Target Encoding IN-FOLD + 5 Seeds _____\n")

SMOOTHING   = 15
GLOBAL_MEAN = y_train.mean()
N_FOLDS     = 5# AXE 3 : 5-fold optimal (std la plus faible)
SEEDS       = [42, 123, 456, 789, 2024]

BASE_PARAMS = {
    "objective":         "reg:squarederror",
    "learning_rate":     0.005,
    "max_depth":         6,
    "subsample":         0.85,
    "colsample_bytree":  0.75,
    "colsample_bylevel": 0.75,
    "colsample_bynode":  0.75,
    "reg_alpha":         0.1,
    "reg_lambda":        2.0,
    "min_child_weight":  8,
    "gamma":             0.0,
    "tree_method":       "hist",
    "n_jobs":            -1,
    "n_estimators":      10000,
}

CV = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(y_train))
test_preds = np.zeros(len(test_merged))
rmse_folds = []

# Target encoding sur TOUT le train pour les prédictions test finales
# (pas de fuite : le test n'a pas de labels)
te_map_full = {}
for c in np.unique(canton_train):
    mask = canton_train == c
    n    = mask.sum()
    m    = y_train[mask].mean()
    te_map_full[c] = (m * n + GLOBAL_MEAN * SMOOTHING) / (n + SMOOTHING)

te_test = np.array([te_map_full.get(c, GLOBAL_MEAN) for c in canton_test])

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train_imp)):

    # Target encoding calculé UNIQUEMENT sur le fold train
    # → le fold val ne contribue pas à son propre encodage → pas de fuite
    canton_tr  = canton_train[tr_idx]
    y_tr       = y_train[tr_idx]
    canton_val = canton_train[val_idx]

    te_map_fold = {}
    for c in np.unique(canton_tr):
        mask = canton_tr == c
        n    = mask.sum()
        m    = y_tr[mask].mean()
        te_map_fold[c] = (m * n + GLOBAL_MEAN * SMOOTHING) / (n + SMOOTHING)

    te_tr  = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_tr])
    te_val = np.array([te_map_fold.get(c, GLOBAL_MEAN) for c in canton_val])

    Xf_tr  = X_train_imp.iloc[tr_idx].copy()
    Xf_val = X_train_imp.iloc[val_idx].copy()
    Xf_tr['canton_te']  = te_tr
    Xf_val['canton_te'] = te_val

    Xf_test = X_test_imp.copy()
    Xf_test['canton_te'] = te_test

    yf_tr  = y_train[tr_idx]
    yf_val = y_train[val_idx]
    wf_tr  = sample_weights[tr_idx]

    # Moyennage sur 5 seeds → réduit la variance stochastique
    fold_val_preds  = np.zeros(len(val_idx))
    fold_test_preds = np.zeros(len(test_merged))

    for seed in SEEDS:
        params_seed = {**BASE_PARAMS, "random_state": seed}
        m = xgb.XGBRegressor(**params_seed, early_stopping_rounds=75, eval_metric="rmse")
        m.fit(Xf_tr, yf_tr,
              sample_weight=wf_tr,
              eval_set=[(Xf_val, yf_val)],
              verbose=False)
        fold_val_preds  += np.clip(m.predict(Xf_val),  0, 100) / len(SEEDS)
        fold_test_preds += np.clip(m.predict(Xf_test), 0, 100) / len(SEEDS)

    oof_preds[val_idx] = fold_val_preds
    test_preds        += fold_test_preds / N_FOLDS

    rmse = np.sqrt(mean_squared_error(yf_val, fold_val_preds))
    rmse_folds.append(rmse)
    print(f"Fold {fold+1}/{N_FOLDS} | RMSE = {rmse:.4f}")

rmse_final = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\nRMSE OOF XGBoost : {rmse_final:.4f}")

bias = float(np.mean(oof_preds - y_train))
predictions = np.clip(test_preds, 0, 100)
if abs(bias) > 0.05:
    predictions = predictions - bias
    print(f"Biais applique : {bias:+.4f}")
print(f"Biais OOF : {bias:+.4f}")


# ________________________________________________________
# Graphiques
# ________________________________________________________

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_train, oof_preds, alpha=0.35, s=10, color='steelblue')
axes[0].scatter(y_train[is_outlier], oof_preds[is_outlier],
                alpha=0.9, s=35, color='red', label='Outliers IQR', zorder=5)
axes[0].plot([0, 100], [0, 100], 'r--', lw=1.5)
axes[0].set_xlabel("% OUI réel")
axes[0].set_ylabel("% OUI prédit")
axes[0].set_title(f"Prédictions OOF vs réel - RMSE = {rmse_final:.4f}")
axes[0].legend(fontsize=8)

errors = y_train - oof_preds
axes[1].hist(errors, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axes[1].axvline(0, color='red', ls='--', lw=2)
axes[1].set_title(f"Distribution des erreurs (biais={bias:+.3f})")
axes[1].set_xlabel("Erreur résiduelle")
axes[1].set_ylabel("Nb communes")

axes[2].bar(range(1, N_FOLDS+1), rmse_folds, color='steelblue', alpha=0.85, edgecolor='black')
axes[2].axhline(rmse_final, color='red', ls='--', lw=2, label=f'Moy={rmse_final:.4f}')
axes[2].set_title("RMSE par fold")
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("RMSE")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("xgb_results.png", dpi=150)
plt.close()

# Feature importance sur tout le train
Xf_full = X_train_imp.copy()
Xf_full['canton_te'] = np.array([te_map_full.get(c, GLOBAL_MEAN) for c in canton_train])
xgb_full = xgb.XGBRegressor(**{**BASE_PARAMS, "random_state": 42})
xgb_full.fit(Xf_full, y_train, sample_weight=sample_weights, verbose=False)
imp_df = pd.DataFrame({
    "feature":    Xf_full.columns,
    "importance": xgb_full.feature_importances_
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top25 = imp_df.head(25)
ax.barh(top25["feature"][::-1], top25["importance"][::-1], color="steelblue", alpha=0.85)
ax.set_xlabel("Importance (gain normalisé)")
ax.set_title("XGBoost — Top 25 features")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
plt.close()

print(f"\nTop 10 features XGB :")
print(imp_df.head(10).to_string(index=False))


# ________________________________________________________
# Soumission
# ________________________________________________________

submission = pd.DataFrame({
    'Id':        test_merged['Id'],
    'Predicted': np.clip(predictions, 0, 100)
})
submission.to_csv('submission_xgboost.csv', index=False)
print("\nSubmission sauvegardee : submission_xgboost.csv")
print(submission.head())
