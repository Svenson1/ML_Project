import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def change_train_dataset(train_m):
    # on supprime car >50% de nan
    train_m = train_m.drop(columns=['PdA/Sol.'])
    train_m = train_m.drop(columns=['PdA/Sol.'])

    # on supprime car colinéarité :
    train_m = train_m.drop(columns=['Settlement and urban area in %'])
    train_m = train_m.drop(columns=['Settlement and urban area in %'])

    # Colonnes identifiants income sans valeur prédictive
    train_m = train_m.drop(columns=['gdenr_income', 'ktnr_income'], errors='ignore')
    train_m = train_m.drop(columns=['gdenr_income', 'ktnr_income'], errors='ignore')

    # gestion des partis politique en nan, nan = parti absent donc valeur = 0
    party_cols = ['SVP', 'SP', 'GPS', 'CVP', 'FDP/PLR 2)', 'GLP', 'BDP',
                  'EVP/CSP', 'Small right-wing parties']
    for col in party_cols:
        if col in train_m.columns:
            train_m[col] = train_m[col].fillna(0)
            train_m[col] = train_m[col].fillna(0)

    return train_m

def create_target(train_m):
    # Define target variable
    y_train = train_m['Ja in Prozent']
    # on encode les Kantons avec one hot :
    dummies_train = pd.get_dummies(train_m['Kantons-Nummer'], prefix='canton',
                                   drop_first=True,
                                   dtype=int)
    dummies_test = pd.get_dummies(train_m['Kantons-Nummer'], prefix='canton',
                                  drop_first=True,
                                  dtype=int)

    # aligner les colonnes
    dummies_train, dummies_test = dummies_train.align(dummies_test, join='left', axis=1, fill_value=0)

    train_merged = pd.concat([train_m, dummies_train], axis=1)
    test_merged = pd.concat([train_m, dummies_test], axis=1)

    train_merged = train_merged.drop(columns=['Kantons-Nummer'])
    test_merged = test_merged.drop(columns=['Kantons-Nummer'])

    return y_train, train_merged, test_merged
