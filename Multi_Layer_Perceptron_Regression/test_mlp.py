import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# ________________________________________________________
# Chargement des Dataset
# ________________________________________________________

#fonction :
def clean_and_format_id(df, column_name):
    """
    :param df: The initial DataFrame
    :param column_name: The name of the column
    :return: the cleaned DataFrame with the format ID
    """
    # 1. Conversion en numérique (force les erreurs en NaN)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    # 2. Suppression des lignes où le numéro de commune est invalide (NaN)
    df = df.dropna(subset=[column_name]).copy()
    # 3. Création de la colonne 'Id' (passage par int pour retirer le .0 si c'est du float, puis str)
    df['Id'] = df[column_name].astype(int).astype(str)
    return df


#training and test set :
train_df = pd.read_csv("../Data_Sets/results_train.csv")
test_df = pd.read_csv("../Data_Sets/results_test.csv")
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)# On ajoute une colone id qui est egale au numero de commune
test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)
train_df = train_df.drop(columns=['Gemeinde-Nummer'])
test_df = test_df.drop(columns=['Gemeinde-Nummer'])


#Other referundum = 622
file_622 = "../Data_Sets/622.00-result-by-canton-district-and-municipality.xlsx"
df_622 = pd.read_excel(file_622, sheet_name="Gemeinden", header=5)
df_622.columns = df_622.columns.str.strip()
df_622 = clean_and_format_id(df_622, 'Gemeinde-Nummer')
df_622 = df_622.drop_duplicates(subset=['Id'])
df_622 = df_622.add_suffix('_622')
df_622 = df_622.rename(columns={'Id_622': 'Id'}) # pour la fusion apres
df_622 = df_622.drop(columns=['Gemeinde-Nummer_622', 'Gemeinde_622', 'Kanton_622'])


#portrait of communes = jee
file_jee = "../Data_Sets/je-e-21.03.01.xlsx"
df_jee = pd.read_excel(file_jee, sheet_name="Schweiz - Gemeinden", header=5)
df_jee = clean_and_format_id(df_jee, 'Number of commune')
df_jee = df_jee.drop_duplicates(subset=['Id'])
df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])
# On force les cols a être des chiffres :
for col in df_jee.columns:
    if col != 'Id':
        df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')


#geoData
file_geo = "../Data_Sets/swiss_communes_geodata.csv"
df_geo = pd.read_csv(file_geo)
df_geo = clean_and_format_id(df_geo, 'bfs_id')
df_geo = df_geo.drop_duplicates(subset=['Id'])
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])



#income data for each Swiss com-mune in 2017
file_income = "../Data_Sets/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
df_income = pd.read_excel(file_income,sheet_name='Gemeinden - Communes')
df_income = clean_and_format_id(df_income, 'gdenr')
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income')
df_income = df_income.rename(columns={'Id_income': 'Id'})
# On force les cols a être des chiffres :
for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')



# ________________________________________________________
# Merge Datasets
# ________________________________________________________

train_merged = (train_df.merge(df_622, on='Id', how='left')
                .merge(df_jee, on='Id', how='left')
                .merge(df_income, on='Id', how='left')
                .merge(df_geo, on='Id', how='left'))

test_merged = (test_df.merge(df_622, on='Id', how='left')
               .merge(df_jee, on='Id', how='left')
               .merge(df_income, on='Id', how='left')
               .merge(df_geo, on='Id', how='left'))

print(f"Doublons dans train_merged : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons dans test_merged  : {test_merged['Id'].duplicated().sum()}")



# ________________________________________________________
# Modifications Dataset après analyse
# ________________________________________________________
#on supprime car >50% de nan
train_merged = train_merged.drop(columns=['PdA/Sol.'])
test_merged = test_merged.drop(columns=['PdA/Sol.'])

#on supprime car colinéarité :
train_merged = train_merged.drop(columns=['Settlement and urban area in %'])
test_merged = test_merged.drop(columns=['Settlement and urban area in %'])

# Colonnes identifiants income sans valeur prédictive
train_merged = train_merged.drop(columns=['gdenr_income', 'ktnr_income'], errors='ignore')
test_merged  = test_merged.drop(columns=['gdenr_income', 'ktnr_income'],  errors='ignore')

# gestion des partis politique en nan, nan = parti absent donc valeur = 0
party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP',
              'EVP/CSP', 'Small right-wing parties']
for col in party_cols:
    if col in train_merged.columns:
        train_merged[col] = train_merged[col].fillna(0)
        test_merged[col]  = test_merged[col].fillna(0)



# ________________________________________________________
# Target
# ________________________________________________________
# Define target variable
y_train = train_merged['Ja in Prozent']
#on encode les Kantons avec one hot :
dummies_train = pd.get_dummies(train_merged['Kantons-Nummer'], prefix='canton',
    drop_first=True,
    dtype=int )
dummies_test  = pd.get_dummies(test_merged['Kantons-Nummer'], prefix='canton',
    drop_first=True,
    dtype=int)

# aligner les colonnes
dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

train_merged = pd.concat([train_merged, dummies_train], axis=1)
test_merged  = pd.concat([test_merged, dummies_test], axis=1)

train_merged = train_merged.drop(columns=['Kantons-Nummer'])
test_merged = test_merged.drop(columns=['Kantons-Nummer'])



# ________________________________________________________
# Selection Des Features
# ________________________________________________________

leakage_columns = [
    'eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
    'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent'
]

# Selection uniquement des colones numériques et on retire le résulat du vote
X_train_raw = train_merged.select_dtypes(include=[np.number]).drop(columns=[c for c in leakage_columns if c in train_merged.columns])

# S'assurer que le test a les mêmes features
X_test_raw = test_merged[X_train_raw.columns]

print(f"Features : {X_train_raw.shape[1]} colonnes")
print(f"NaN dans X_train : {X_train_raw.isna().sum().sum()}")
print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")



# ________________________________________________________
# Pipeline avec les meilleurs paramètres connus
# ________________________________________________________
# Meilleurs paramètres issus du grid search précédent :
# rank 3 — meilleur compromis RMSE/std :
# selector__k=80, hidden_layer_sizes=(128,), alpha=5,
# learning_rate_init=0.0005, batch_size=32

pipeline_for_bagging = Pipeline([
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



# ________________________________________________________
# Bagging
# ________________________________________________________

bagged_model = BaggingRegressor(
    estimator=pipeline_for_bagging,
    n_estimators=25,
    max_samples=0.85,
    bootstrap=True,
    random_state=42,
    n_jobs=1,
    verbose=1,
)

print("\nEntraînement du BaggingRegressor (25 estimateurs)...")
bagged_model.fit(X_train_raw, y_train)
print("Bagging terminé.")

# ________________________________________________________
# Evaluation CV locale
# ________________________________________________________

cv_scores = cross_val_score(
    bagged_model,
    X_train_raw, y_train,
    cv=13,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,  # ← 1 car bagged_model est déjà parallèle
)

rmse_scores = -cv_scores
print(f"\nRMSE CV moyen : {rmse_scores.mean():.3f}")
print(f"Std CV        : {rmse_scores.std():.3f}")


# ________________________________________________________
# Soumission
# ________________________________________________________

predictions = np.clip(bagged_model.predict(X_test_raw), 0, 100)
submission = pd.DataFrame({
    'Id': test_merged['Id'],
    'Predicted': predictions
})
submission.to_csv('submission_mlp_test.csv', index=False)
print("Submission sauvegardée.")
print(submission.head())



