import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# ________________________________________________________
# Auto-detection du chemin (local ou Kaggle)
# ________________________________________________________

def find_dataset_dir():
    if not os.path.exists("/kaggle/input"):
        return "../Data-Sets"
    competitions_path = "/kaggle/input/competitions"
    if os.path.exists(competitions_path):
        for folder in os.listdir(competitions_path):
            folder_path = os.path.join(competitions_path, folder)
            if os.path.isdir(folder_path) and "results_train.csv" in os.listdir(folder_path):
                return folder_path
    for folder in os.listdir("/kaggle/input"):
        folder_path = os.path.join("/kaggle/input", folder)
        if os.path.isdir(folder_path) and folder != "competitions":
            if "results_train.csv" in os.listdir(folder_path):
                return folder_path
    raise FileNotFoundError("Dataset introuvable.")

def clean_and_format_id(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name]).copy()
    df['Id'] = df[column_name].astype(int).astype(str)
    return df


# ________________________________________________________
# Chargement des datasets
# ________________________________________________________

DATA_DIR = find_dataset_dir()

train_df = pd.read_csv(f"{DATA_DIR}/results_train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/results_test.csv")
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(int).astype(str)
test_df['Id']  = test_df['Gemeinde-Nummer'].astype(int).astype(str)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# AXE 1 — ANTI-LEAKAGE
# Le fichier 622 contient 2240 communes dont certaines n'existent ni dans
# train ni dans test. Les merger introduit du bruit et cause un distribution
# shift massif (6.30 au lieu de 5.92 en public).
# Fix : ne merger QUE les IDs qui existent dans train+test, calculé avant tout merge.
valid_ids = set(train_df['Id'].unique()) | set(test_df['Id'].unique())
print(f"Valid IDs (train+test): {len(valid_ids)}")

df_622_raw = pd.read_excel(
    f"{DATA_DIR}/622.00-result-by-canton-district-and-municipality.xlsx",
    sheet_name="Gemeinden", header=5
)
df_622_raw.columns = df_622_raw.columns.str.strip()
df_622_raw["Gemeinde-Nummer"] = pd.to_numeric(df_622_raw["Gemeinde-Nummer"], errors="coerce")
df_622 = df_622_raw[df_622_raw["Gemeinde-Nummer"] >= 1000].copy()
df_622 = clean_and_format_id(df_622, "Gemeinde-Nummer")
df_622 = df_622.drop_duplicates(subset=["Id"])
df_622 = df_622[df_622['Id'].isin(valid_ids)].copy()
print(f"622 apres filtre anti-leakage : {len(df_622)} communes")

# On garde uniquement la structure electorale (participation, electeurs)
# Les resultats du vote 622 ne sont pas correles avec le vote 623
df_622 = df_622.drop(columns=["Gemeinde-Nummer", "Gemeinde", "Kanton",
                               "Ja-Stimmen", "Nein-Stimmen", "Ja in Prozent"], errors="ignore")
df_622 = df_622.add_suffix("_622")
df_622 = df_622.rename(columns={"Id_622": "Id"})

df_jee = pd.read_excel(f"{DATA_DIR}/je-e-21.03.01.xlsx",
                        sheet_name="Schweiz - Gemeinden", header=5)
df_jee = clean_and_format_id(df_jee, "Number of commune")
df_jee = df_jee.drop_duplicates(subset=["Id"])
df_jee = df_jee.drop(columns=["Number of commune", "Name of commune"], errors="ignore")
for col in df_jee.columns:
    if col != "Id":
        df_jee[col] = pd.to_numeric(df_jee[col], errors="coerce")

df_geo = pd.read_csv(f"{DATA_DIR}/swiss_communes_geodata.csv")
df_geo = clean_and_format_id(df_geo, "bfs_id")
df_geo = df_geo.drop_duplicates(subset=["Id"])
df_geo = df_geo.drop(columns=["bfs_id", "municipalityLabel"], errors="ignore")

df_income = pd.read_excel(f"{DATA_DIR}/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx",
                           sheet_name="Gemeinden - Communes")
df_income = clean_and_format_id(df_income, "gdenr")
df_income = df_income.drop_duplicates(subset=["Id"])
df_income = df_income.drop(columns=["ktname", "gdename", "Einheit"], errors="ignore")
df_income = df_income.add_suffix("_income")
df_income = df_income.rename(columns={"Id_income": "Id"})
for col in df_income.columns:
    if col != "Id":
        df_income[col] = pd.to_numeric(df_income[col], errors="coerce")


# ________________________________________________________
# Merge Datasets
# ________________________________________________________

train_merged = (train_df.merge(df_622,    on="Id", how="left")
                        .merge(df_jee,    on="Id", how="left")
                        .merge(df_income, on="Id", how="left")
                        .merge(df_geo,    on="Id", how="left"))

test_merged  = (test_df.merge(df_622,    on="Id", how="left")
                       .merge(df_jee,    on="Id", how="left")
                       .merge(df_income, on="Id", how="left")
                       .merge(df_geo,    on="Id", how="left"))

print(f"Doublons train : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons test  : {test_merged['Id'].duplicated().sum()}")
print(f"Shape : train={train_merged.shape} | test={test_merged.shape}")


# ________________________________________________________
# Nettoyage
# ________________________________________________________

train_merged = train_merged.drop(columns=['PdA/Sol.'], errors='ignore')
test_merged  = test_merged.drop(columns=['PdA/Sol.'],  errors='ignore')

for col in ['Settlement and urban area in %', 'gdenr_income', 'ktnr_income']:
    train_merged = train_merged.drop(columns=[col], errors='ignore')
    test_merged  = test_merged.drop(columns=[col],  errors='ignore')

party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP',
              'EVP/CSP', 'Small right-wing parties']
for col in party_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col]  = test_merged[col].fillna(0)


# ________________________________________________________
# Encodage des cantons (one-hot)
# ________________________________________________________

kanton_col = next((c for c in ["Kantons-Nummer_622", "Kantons-Nummer"]
                   if c in train_merged.columns), None)

if kanton_col:
    dummies_train = pd.get_dummies(train_merged[kanton_col], prefix='canton',
                                   drop_first=True, dtype=int)
    dummies_test  = pd.get_dummies(test_merged[kanton_col],  prefix='canton',
                                   drop_first=True, dtype=int)
    dummies_train, dummies_test = dummies_train.align(dummies_test, join='left',
                                                       axis=1, fill_value=0)
    train_merged = pd.concat([train_merged.drop(columns=[kanton_col]), dummies_train], axis=1)
    test_merged  = pd.concat([test_merged.drop(columns=[kanton_col]),  dummies_test],  axis=1)


# ________________________________________________________
# AXE 2 — FEATURE ENGINEERING MINIMAL (Anti Distribution Shift)
# ________________________________________________________
# Probleme : les 80+ features engineered (agri_x_jeunes, rural_score, etc.)
# ont une correlation elevee sur le train (r=0.63) mais overfit massivement.
# Le test public a une distribution differente — ces interactions ne generalisent pas.
# Solution : garder UNIQUEMENT les features exogenes stables et 2 transformations simples.

def add_features(df):
    df = df.copy()

    pop_col = next((c for c in df.columns
                    if any(kw in c.lower() for kw in ["resident", "einwohn"])
                    and pd.to_numeric(df[c], errors='coerce').median() > 100), None)

    if pop_col:
        pop = pd.to_numeric(df[pop_col], errors='coerce').fillna(100).clip(lower=1)
        # log_population : transformation stable, generalise bien
        df["log_population"] = np.log1p(pop)
        # SUPPRIME : vote_fiabilite, taille_commune, agri_x_log_pop

    if "Stimmberechtigte_622" in df.columns:
        elec = pd.to_numeric(df["Stimmberechtigte_622"], errors='coerce').fillna(100).clip(lower=1)
        df["log_electeurs"] = np.log1p(elec)
        # SUPPRIME : fiabilite_electorale

    # ratio_agri_emploi : ratio stable, pas specifique au train
    if "Agricultural employment" in df.columns and "Total employment" in df.columns:
        df["ratio_agri_emploi"] = df["Agricultural employment"] / (df["Total employment"] + 1)

    # agri_area_sq : seule transformation non-lineaire conservee
    # Les autres (cbrt, interactions) sont trop specifiques
    if "Agricultural area in %" in df.columns:
        df["agri_area_sq"] = df["Agricultural area in %"] ** 2

    # SUPPRIME TOUT CE QUI EST INTERACTION :
    # agri_x_jeunes, agri_x_menages, rural_score, agri_x_seniors, agri_x_improd
    # svp_x_agri, fdp_x_agri, cvp_x_agri, droite_x_agri, droite_totale
    # romand_x_agri, aleman_x_agri, is_romand, is_tessin, is_aleman
    # participation_x_agri

    return df

train_merged = add_features(train_merged)
test_merged  = add_features(test_merged)


# ________________________________________________________
# Preparation features
# ________________________________________________________

leakage_columns = [
    "eingelegte Stimmzettel", "Stimmbeteiligung", "leere Stimmzettel",
    "ungultige Stimmzettel", "gultige Stimmen", "Ja-Stimmen", "Nein-Stimmen",
    "Ja in Prozent", "Gemeinde-Nummer", "Stimmberechtigte",
    "ungültige Stimmzettel", "gültige Stimmen",
]

y_train = train_merged["Ja in Prozent"].values

X_train_raw = train_merged.select_dtypes(include=[np.number]).drop(
    columns=[c for c in leakage_columns if c in train_merged.columns]
)
X_test_raw = test_merged[X_train_raw.columns]

missing_ratio = X_train_raw.isna().mean()
X_train_raw = X_train_raw.drop(columns=missing_ratio[missing_ratio > 0.6].index.tolist())
X_test_raw  = X_test_raw[X_train_raw.columns]

print(f"Features : {X_train_raw.shape[1]} colonnes (minimal, stable)")
print(f"NaN dans X_train : {X_train_raw.isna().sum().sum()}")

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw),  columns=X_test_raw.columns)

# AXE 3 — SUPPRIMER LES SAMPLE WEIGHTS
# Les sample_weights log(population) sur-penalisent les petites communes.
# Le test public en contient probablement plus que le train → erreur amplifiee.
# Poids uniformes = chaque commune compte autant = meilleure generalisation.
sample_weights = np.ones(len(y_train))
print("Sample weights : uniformes (pas de ponderation population)")


# ________________________________________________________
# XGBoost — Early stopping pour trouver n_estimators
# ________________________________________________________

CV5 = KFold(n_splits=5, shuffle=True, random_state=42)

# AXE 4 — REGULARISATION REDUITE
# Avant : reg_alpha=0.1, reg_lambda=2.0 → sous-fit sur le test public
# Apres : reg_alpha=0.05, reg_lambda=1.0 → meilleure generalisation
BASE_PARAMS = {
    "objective":         "reg:squarederror",
    "learning_rate":     0.003,
    "max_depth":         5,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "colsample_bylevel": 0.7,
    "reg_alpha":         0.05,
    "reg_lambda":        1.0,
    "min_child_weight":  3,
    "gamma":             0.1,
    "random_state":      42,
    "tree_method":       "hist",
    "n_jobs":            -1,
}

best_n_est_list = []
for fold, (tr_idx, val_idx) in enumerate(CV5.split(X_train_imp)):
    m = xgb.XGBRegressor(**BASE_PARAMS, n_estimators=15000,
                         early_stopping_rounds=400, eval_metric="rmse")
    m.fit(X_train_imp.iloc[tr_idx], y_train[tr_idx],
          eval_set=[(X_train_imp.iloc[val_idx], y_train[val_idx])],
          verbose=False)
    best_n_est_list.append(m.best_iteration)
    preds = np.clip(m.predict(X_train_imp.iloc[val_idx]), 0, 100)
    rmse  = np.sqrt(mean_squared_error(y_train[val_idx], preds))
    print(f"Fold {fold+1}/5 : n_est = {m.best_iteration:6d}  RMSE = {rmse:.4f}")

n_est_opt = int(np.mean(best_n_est_list))
print(f"\nn_estimators optimal : {n_est_opt}")


# ________________________________________________________
# XGBoost — Recherche hyperparametres
# ________________________________________________________

param_grid = {
    "max_depth":         [4, 5, 6],
    "learning_rate":     [0.002, 0.003, 0.005],
    "subsample":         [0.7, 0.75, 0.8, 0.85],
    "colsample_bytree":  [0.6, 0.65, 0.7, 0.75],
    "colsample_bylevel": [0.6, 0.65, 0.7],
    "reg_alpha":         [0.01, 0.05, 0.1],
    "reg_lambda":        [0.5, 1.0, 1.5],
    "min_child_weight":  [2, 3, 4],
    "gamma":             [0.05, 0.1, 0.15],
}

search = RandomizedSearchCV(
    xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                     tree_method="hist", n_jobs=-1, n_estimators=n_est_opt),
    param_distributions=param_grid,
    n_iter=60, cv=CV5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, random_state=42, verbose=1, refit=True,
)
search.fit(X_train_imp, y_train)

best_params  = search.best_params_
best_cv_rmse = -search.best_score_
print(f"\nMeilleurs parametres : {best_params}")
print(f"Meilleur RMSE CV    : {best_cv_rmse:.3f}")


# ________________________________________________________
# Evaluation finale OOF + predictions test
# ________________________________________________________

final_params    = {**BASE_PARAMS, **best_params}
oof_preds_train = np.zeros(len(y_train))
test_preds_avg  = np.zeros(len(test_merged))
oof_rmse_final  = []

for fold, (tr_idx, val_idx) in enumerate(CV5.split(X_train_imp)):
    m = xgb.XGBRegressor(**final_params, n_estimators=n_est_opt)
    m.fit(X_train_imp.iloc[tr_idx], y_train[tr_idx], verbose=False)

    p_val = np.clip(m.predict(X_train_imp.iloc[val_idx]), 0, 100)
    oof_preds_train[val_idx] = p_val
    test_preds_avg += np.clip(m.predict(X_test_imp), 0, 100) / 5

    rmse = np.sqrt(mean_squared_error(y_train[val_idx], p_val))
    oof_rmse_final.append(rmse)

print(f"\nRMSE par fold : {np.round(oof_rmse_final, 4)}")
print(f"RMSE moyen    : {np.mean(oof_rmse_final):.4f} +/- {np.std(oof_rmse_final):.4f}")

# AXE 5 — CALIBRATION POST-PREDICTION
# Si le modele a un biais systematique (predit toujours trop haut ou trop bas),
# on le detecte sur les predictions OOF et on le corrige sur le test.
# Exemple : biais = +1.5 → toutes les predictions sont 1.5 points trop hautes.
bias = float(np.mean(oof_preds_train - y_train))
print(f"Biais OOF detecte : {bias:+.3f} points")
if abs(bias) > 0.1:
    test_preds_avg = np.clip(test_preds_avg - bias, 0, 100)
    print(f"Calibration appliquee : correction de {bias:+.3f} points")
else:
    print("Biais negligeable, pas de calibration appliquee")


# ________________________________________________________
# Graphiques pour le rapport
# ________________________________________________________

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_train, oof_preds_train, alpha=0.35, s=10, color="steelblue")
ax.plot([0, 100], [0, 100], "r--", lw=1.5)
ax.set_xlabel("Vrai % Ja in Prozent")
ax.set_ylabel("Predit % Ja in Prozent")
ax.set_title(f"XGBoost - RMSE CV = {np.mean(oof_rmse_final):.3f}")
ax.set_xlim([0, 100]); ax.set_ylim([0, 100])
plt.tight_layout(); plt.savefig("xgb_pred_vs_true.png", dpi=150); plt.close()

xgb_final = xgb.XGBRegressor(**final_params, n_estimators=n_est_opt + 100)
xgb_final.fit(X_train_imp, y_train, verbose=False)

imp_df = pd.DataFrame({"feature": X_train_imp.columns,
                        "importance": xgb_final.feature_importances_}
                      ).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(9, 7))
top_n = imp_df.head(min(20, len(imp_df)))
ax.barh(top_n["feature"][::-1], top_n["importance"][::-1], color="steelblue", alpha=0.85)
ax.set_xlabel("Importance (gain normalise)")
ax.set_title("XGBoost - Top features")
plt.tight_layout(); plt.savefig("xgb_feature_importance.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([f"Fold {i+1}" for i in range(5)], oof_rmse_final, color="steelblue", alpha=0.85)
ax.axhline(np.mean(oof_rmse_final), color="red", ls="--", lw=1.5,
           label=f"Moyenne = {np.mean(oof_rmse_final):.4f}")
ax.set_ylabel("RMSE"); ax.set_title("XGBoost - RMSE par fold")
ax.legend(); plt.tight_layout()
plt.savefig("xgb_rmse_par_fold.png", dpi=150); plt.close()


# ________________________________________________________
# Soumission
# ________________________________________________________

predictions = np.clip(test_preds_avg, 0, 100)
submission = pd.DataFrame({
    'Id':        test_merged['Id'],
    'Predicted': predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission sauvegardee.")
print(submission.head())

print(f"""
=================================================================
RESUME EXPERT XGB
=================================================================
  AXE 1 - Anti-leakage 622    : {len(df_622)} communes (filtre valid_ids)
  AXE 2 - Features minimales  : {X_train_imp.shape[1]} colonnes (pas d'interactions)
  AXE 3 - Sample weights      : SUPPRIMES (poids uniformes)
  AXE 4 - Regularisation      : alpha={final_params.get('reg_alpha')}, lambda={final_params.get('reg_lambda')} (reduite)
  AXE 5 - Calibration         : {bias:+.3f} points corriges
  RMSE OOF                    : {np.mean(oof_rmse_final):.4f}
=================================================================
""")