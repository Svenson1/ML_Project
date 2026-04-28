import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#fonction :
def clean_and_format_id(df, column_name):
    # 1. Conversion en numérique (force les erreurs en NaN)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    # 2. Suppression des lignes où le numéro de commune est invalide (NaN)
    df = df.dropna(subset=[column_name]).copy()
    # 3. Création de la colonne 'Id' (passage par int pour retirer le .0 si c'est du float, puis str)
    df['Id'] = df[column_name].astype(int).astype(str)
    return df


#training and test set :
train_df = pd.read_csv("../Data-Sets/results_train.csv")
test_df = pd.read_csv("../Data-Sets/results_test.csv")

# On ajoute une colone id qui est egale au numero de commune
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)

#on load : Other referundum -------------------
file_622 = "../Data-Sets/622.00-result-by-canton-district-and-municipality.xlsx"
df_622 = pd.read_excel(file_622, sheet_name="Gemeinden", header=5)
df_622.columns = df_622.columns.str.strip()

df_622 = clean_and_format_id(df_622, 'Gemeinde-Nummer')

df_622 = df_622.add_suffix('_622')
df_622 = df_622.rename(columns={'Id_622': 'Id'}) # pour la fusion apres
df_622 = df_622.drop(columns=['Gemeinde-Nummer_622', 'Gemeinde_622', 'Kanton_622'])

#on load : "portrait of communes" = jee
file_jee = "../Data-Sets/je-e-21.03.01.xlsx"
df_jee = pd.read_excel(file_jee, sheet_name="Schweiz - Gemeinden", header=5)

#clean id
df_jee = clean_and_format_id(df_jee, 'Number of commune')
df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])


# On force les cols a être des chiffres :
for col in df_jee.columns:
    if col != 'Id':
        df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')


#on load: geoData:--------------
file_geo = "../Data-Sets/swiss_communes_geodata.csv"
df_geo = pd.read_csv(file_geo)

#clean id
df_geo = clean_and_format_id(df_geo, 'bfs_id')
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

#on load : income data for each Swiss com-mune in 2017-----------
file_income = "../Data-Sets/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
df_income = pd.read_excel(file_income,sheet_name='Gemeinden - Communes')

df_income = clean_and_format_id(df_income, 'gdenr')
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income')
df_income = df_income.rename(columns={'Id_income': 'Id'})

for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')




# Merge datasets
train_merged = train_df.merge(df_622, on='Id', how='left').merge(df_jee, on='Id', how='left').merge(df_income, on='Id', how='left').merge(df_geo, on='Id', how='left')
test_merged = test_df.merge(df_622, on='Id', how='left').merge(df_jee, on='Id', how='left').merge(df_income, on='Id', how='left').merge(df_geo, on='Id', how='left')

# Define target variable
y_train = train_merged['Ja in Prozent']

leakage_columns = [
    'eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
    'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen', 'Ja in Prozent'
]

# Select only number columns and remove voting results
X_train_raw = train_merged.select_dtypes(include=[np.number]).drop(columns=[c for c in leakage_columns if c in train_merged.columns])

# Ensure test set has exactly the same feature columns
X_test_raw = test_merged[X_train_raw.columns]

print(f"Features : {X_train_raw.shape[1]} colonnes")
print(f"NaN dans X_train : {X_train_raw.isna().sum().sum()}")
print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

# Impute missing values (replace NaNs with mean of the column)
#On fait ça via un pipeline :
#scaler obligatoire pour mlp
pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()),
                     ('mlp', MLPRegressor(
                         hidden_layer_sizes=(128, 64),
                         activation="relu",
                         solver="adam",
                         max_iter=1000,
                         random_state=42,
                         early_stopping=True,
                         validation_fraction=0.1,
                         n_iter_no_change=10,
                     ))])

#Eval --------------
scores = cross_val_score(pipeline, X_train_raw, y_train, cv=5, scoring='neg_root_mean_squared_error')

print(f"\nRMSE CV : {-scores.mean():.3f} ± {scores.std():.3f}")



