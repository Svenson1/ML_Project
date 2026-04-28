import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
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
df_622 = df_622.drop_duplicates(subset=['Id'])

df_622 = df_622.add_suffix('_622')
df_622 = df_622.rename(columns={'Id_622': 'Id'}) # pour la fusion apres
df_622 = df_622.drop(columns=['Gemeinde-Nummer_622', 'Gemeinde_622', 'Kanton_622'])

#on load : "portrait of communes" = jee
file_jee = "../Data-Sets/je-e-21.03.01.xlsx"
df_jee = pd.read_excel(file_jee, sheet_name="Schweiz - Gemeinden", header=5)

#clean id
df_jee = clean_and_format_id(df_jee, 'Number of commune')
df_jee = df_jee.drop_duplicates(subset=['Id'])
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
df_geo = df_geo.drop_duplicates(subset=['Id'])
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

#on load : income data for each Swiss com-mune in 2017-----------
file_income = "../Data-Sets/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
df_income = pd.read_excel(file_income,sheet_name='Gemeinden - Communes')

df_income = clean_and_format_id(df_income, 'gdenr')
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income')
df_income = df_income.rename(columns={'Id_income': 'Id'})

for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')




# Merge datasets
train_merged = train_df.merge(df_622, on='Id', how='left').merge(df_jee, on='Id', how='left').merge(df_income, on='Id', how='left').merge(df_geo, on='Id', how='left')
test_merged = test_df.merge(df_622, on='Id', how='left').merge(df_jee, on='Id', how='left').merge(df_income, on='Id', how='left').merge(df_geo, on='Id', how='left')

print(f"Doublons dans train_merged : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons dans test_merged  : {test_merged['Id'].duplicated().sum()}")


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
                     ('selector', SelectKBest(score_func=f_regression)),
                     ('mlp', MLPRegressor(
                         hidden_layer_sizes=(128, 64),
                         activation="relu",
                         solver="adam",
                         max_iter=2000,
                         random_state=42,
                         early_stopping=True,
                         validation_fraction=0.1,
                         n_iter_no_change=5,
                     ))])

#Eval -------------- With grid search
param_grid = {
    'selector__k': [20, 30, 40, 50],
    'mlp__hidden_layer_sizes': [
        (128,),         # 1 couche moyenne
        (128, 64),      # 2 couches (ta config actuelle)
        (128, 64, 32),  # 3 couches
        (256, 128),
    ],
    'mlp__activation': ['tanh'],
    'mlp__solver': ['sgd'],
    'mlp__alpha': [0.1, 0.5, 1.0],
    'mlp__learning_rate_init': [0.0005, 0.001, 0.005],
}


grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,n_iter= 200,  cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_raw, y_train)

print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur RMSE CV    : {-grid_search.best_score_:.3f}")


# Afficher le top 5 des configurations
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')
top5 = results[['params', 'mean_test_score', 'std_test_score']].head(5).copy()
top5['RMSE'] = -top5['mean_test_score']
top5['std']  =  top5['std_test_score']
print("\nTop 5 configurations :")
print(top5[['params', 'RMSE', 'std']].to_string(index=False))


# Le best_estimator_ est déjà fitté sur tout X_train_raw
best_pipeline = grid_search.best_estimator_
predictions = np.clip(best_pipeline.predict(X_test_raw), 0, 100) #on clip pour rester entre 0-100

submission = pd.DataFrame({
    'Id': test_merged['Id'],
    'Predicted': predictions
})
submission.to_csv('submission_mlp.csv', index=False)
print("Submission sauvegardée.")

print(submission.head())



