import pandas as pd
import numpy as np

from Multi_Layer_Perceptron_Regression.mlp_utils import change_train_dataset, create_target
from Data_Sets.dataset_utils import import_and_merge_dataset

np.random.seed(42)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# ________________________________________________________
# Chargement  et merge des Dataset
# ________________________________________________________

train_merged, test_merged = import_and_merge_dataset()

print(f"Doublons dans train_merged : {train_merged['Id'].duplicated().sum()}")
print(f"Doublons dans test_merged  : {test_merged['Id'].duplicated().sum()}")



# ________________________________________________________
# Modifications Dataset après analyse
# ________________________________________________________

train_merged, test_merged = change_train_dataset(train_merged, test_merged)



# ________________________________________________________
# Target
# ________________________________________________________
y_train, train_merged, test_merged = create_target(train_merged, test_merged)



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
# Pipeline MLP
# ________________________________________________________

# Impute missing values (replace NaNs with mean of the column)
#scaler obligatoire pour mlp
def mlp_pipeline():
    return Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('selector', SelectKBest(score_func=lambda X, y: mutual_info_regression(X, y, random_state=42))),
    ('mlp',      MLPRegressor(
        max_iter=4000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )),
])

pipeline = mlp_pipeline()


# ________________________________________________________
# Eval avec grid_search
# ________________________________________________________
def mlp_param():
    return {
    # Feature selection
    'selector__k': [50,60,70,80,'all'],
    # Architecture du réseau
    'mlp__hidden_layer_sizes': [
        (128,),
        (256,),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ],
    # Activation
    'mlp__activation': ['tanh'],
    'mlp__solver': ['adam'],
     # Régularisation
    'mlp__alpha': [
        3,
        5,
    ],
    # Learning rate
    'mlp__learning_rate_init': [
        0.0005,
        0.001,
        0.002
    ],
    # Batch size
    'mlp__batch_size': [
        32,
        64,
        128
    ],
}

param_grid = mlp_param()


grid_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,
    cv=13,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=2,
)
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



# ________________________________________________________
# Features Sélectionnées
# ________________________________________________________
# Le best_estimator_ est déjà fitté sur tout X_train_raw
best_pipeline = grid_search.best_estimator_
selector      = best_pipeline.named_steps['selector']
k             = grid_search.best_params_['selector__k']

if k != 'all':
    selected = X_train_raw.columns[selector.get_support()].tolist()
    scores   = selector.scores_[selector.get_support()]
    feat_df  = pd.DataFrame({'feature': selected, 'score': scores})
    feat_df  = feat_df.sort_values('score', ascending=False)
    print(feat_df.to_string(index=False))

# ________________________________________________________
# Soumission
# ________________________________________________________
predictions = np.clip(best_pipeline.predict(X_test_raw), 0, 100) #on clip pour rester entre 0-100
submission = pd.DataFrame({
    'Id': test_merged['Id'],
    'Predicted': predictions
})
submission.to_csv('submission_mlp.csv', index=False)
print("Submission sauvegardée.")

print(submission.head())


