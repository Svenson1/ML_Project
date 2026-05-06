import pandas as pd
import numpy as np

from Multi_Layer_Perceptron_Regression.mlp_utils import change_train_dataset, create_target
from Data_Sets.dataset_utils import import_and_merge_dataset

np.random.seed(42)
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# ________________________________________________________
# Chargement et merge des Dataset
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
# Define target variable
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
    n_jobs=1,  # <- 1 car bagged_model est déjà parallèle
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



