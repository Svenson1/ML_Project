"""
============================================================
EDA COMPLET – Référendum "Vache à Cornes" 2018
============================================================
Script unique regroupant :
  1.  Chargement & merge de tous les datasets
  2.  Variable cible (Ja in Prozent) – scatter communes + distribution
  3.  Valeurs manquantes après merge
  4.  Corrélations (sans leakage)
  5.  Détection d'anomalies et de patterns
  6.  Rapport console récapitulatif

Usage : python eda_complet.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "../Data-Sets/"
TARGET   = "Ja in Prozent"

# Colonnes de leakage : ce sont d'AUTRES résultats du MÊME référendum.
# Les garder ferait "tricher" le modèle car elles permettent de recalculer
# exactement la cible. On les retire avant toute analyse de corrélation.
LEAKAGE_COLS = [
    'eingelegte Stimmzettel', 'Stimmbeteiligung', 'leere Stimmzettel',
    'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen',
]

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


#
# 1.  CHARGEMENT & MERGE
#
section("1. CHARGEMENT & MERGE DES DATASETS")

def clean_id(df, col):
    """Convertit la colonne identifiant commune en string propre 'Id'."""
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col]).copy()
    df['Id'] = df[col].astype(int).astype(str)
    return df

# Train / Test
train_df = pd.read_csv(DATA_DIR + "results_train.csv")
test_df  = pd.read_csv(DATA_DIR + "results_test.csv")
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id']  = test_df['Gemeinde-Nummer'].astype(str)

# Référendum précédent (622)
df_622 = pd.read_excel(
    DATA_DIR + "622.00-result-by-canton-district-and-municipality.xlsx",
    sheet_name="Gemeinden", header=5)
df_622.columns = df_622.columns.str.strip()
df_622 = clean_id(df_622, 'Gemeinde-Nummer')
df_622 = df_622.drop_duplicates(subset=['Id'])
df_622 = df_622.add_suffix('_622').rename(columns={'Id_622': 'Id'})
df_622 = df_622.drop(columns=['Gemeinde-Nummer_622', 'Gemeinde_622', 'Kanton_622'])

# Portrait des communes (jee)
df_jee = pd.read_excel(DATA_DIR + "je-e-21.03.01.xlsx",
                       sheet_name="Schweiz - Gemeinden", header=5)
df_jee = clean_id(df_jee, 'Number of commune')
df_jee = df_jee.drop_duplicates(subset=['Id'])
df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])
for col in df_jee.columns:
    if col != 'Id':
        df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')

# Géodonnées
df_geo = pd.read_csv(DATA_DIR + "swiss_communes_geodata.csv")
df_geo = clean_id(df_geo, 'bfs_id')
df_geo = df_geo.drop_duplicates(subset=['Id'])
df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

# Revenus
df_income = pd.read_excel(
    DATA_DIR + "statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx",
    sheet_name='Gemeinden - Communes')
df_income = clean_id(df_income, 'gdenr')
df_income = df_income.drop_duplicates(subset=['Id'])
df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
df_income = df_income.add_suffix('_income').rename(columns={'Id_income': 'Id'})
for col in df_income.columns:
    if col != 'Id':
        df_income[col] = pd.to_numeric(df_income[col], errors='coerce')

# Merge LEFT JOIN sur l'identifiant commune
def do_merge(base):
    return (base
            .merge(df_622,    on='Id', how='left')
            .merge(df_jee,    on='Id', how='left')
            .merge(df_income, on='Id', how='left')
            .merge(df_geo,    on='Id', how='left'))

train_m = do_merge(train_df)
test_m  = do_merge(test_df)


print(f"  train_merged : {train_m.shape[0]} lignes x {train_m.shape[1]} colonnes")
print(f"  test_merged  : {test_m.shape[0]}  lignes x {test_m.shape[1]} colonnes")
print(f"  Doublons Id train : {train_m['Id'].duplicated().sum()}")
print(f"  Doublons Id test  : {test_m['Id'].duplicated().sum()}")


#
# 2.  VARIABLE CIBLE
#
section("2. VARIABLE CIBLE : Ja in Prozent")

y = train_m[TARGET].dropna()

# Statistiques descriptives
print(f"\n  Nombre de communes (train) : {len(y)}")
print(f"  Minimum   : {y.min():.2f}%")
print(f"  Maximum   : {y.max():.2f}%")
print(f"  Moyenne   : {y.mean():.2f}%")
print(f"  Mediane   : {y.median():.2f}%")
print(f"  Ecart-type: {y.std():.2f}")

# Outliers (méthode IQR)
# L'IQR (Interquartile Range) est la distance entre le 1er quartile (Q1=25%)
# et le 3e quartile (Q3=75%). C'est une mesure robuste de la dispersion.
# Un point est considéré outlier s'il est EN DEHORS de la "cloture" :
#   borne basse = Q1 - 1.5 x IQR
#   borne haute = Q3 + 1.5 x IQR
# Ce sont exactement les moustaches du boxplot.
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR    = Q3 - Q1
lo     = Q1 - 1.5 * IQR
hi     = Q3 + 1.5 * IQR

outliers = train_m[(train_m[TARGET] < lo) | (train_m[TARGET] > hi)].copy()

print(f"\n  Outliers IQR :")
print(f"    Q1 = {Q1:.2f}%  |  Q3 = {Q3:.2f}%  |  IQR = {IQR:.2f}")
print(f"    Borne basse = Q1 - 1.5 x IQR = {lo:.2f}%")
print(f"    Borne haute = Q3 + 1.5 x IQR = {hi:.2f}%")
print(f"    -> {len(outliers)} communes outliers ({len(outliers)/len(y)*100:.1f}%)")


if len(outliers):
    print(f"\n  Communes outliers (triees par % OUI) :")
    print(outliers[["Gemeinde", TARGET]].sort_values(TARGET).to_string(index=False))

# ── Figure 1 : Scatter – numéro de commune vs % de OUI ──────────────────────
# On garde le numero de commune sur l'axe X (ordre original, non trie) pour
# pouvoir identifier quelle commune correspond a quel pourcentage.
# Les outliers IQR sont mis en evidence en rouge.
fig, ax = plt.subplots(figsize=(14, 5))

normal_mask = (train_m[TARGET] >= lo) & (train_m[TARGET] <= hi) & train_m[TARGET].notna()
ax.scatter(train_m.loc[normal_mask, 'Gemeinde-Nummer'],
           train_m.loc[normal_mask, TARGET],
           s=8, alpha=0.5, color='#3498db', label='Communes normales')

if len(outliers):
    ax.scatter(outliers['Gemeinde-Nummer'], outliers[TARGET],
               s=25, alpha=0.9, color='#e74c3c',
               label=f'Outliers IQR (n={len(outliers)})', zorder=5)

ax.axhline(y.mean(),   color='red',    linestyle='--', linewidth=1.5,
           label=f"Moyenne = {y.mean():.1f}%")
ax.axhline(50,         color='gray',   linestyle=':',  linewidth=1.2,
           label="50%")
ax.axhline(hi,         color='orange', linestyle='--', linewidth=1,
           label=f"Borne haute IQR = {hi:.1f}%")
ax.axhline(lo,         color='orange', linestyle='--', linewidth=1,
           label=f"Borne basse IQR = {lo:.1f}%")

ax.set_xlabel("Numero de commune (Gemeinde-Nummer)")
ax.set_ylabel("Ja in Prozent (%)")
ax.set_title("% de OUI par commune – numero original (outliers IQR en rouge)")
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("eda_fig_01_communes_scatter.png", dpi=150)
plt.show()

# Figure 2 : Histogramme + Boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Distribution de 'Ja in Prozent'", fontsize=13, fontweight='bold')

axes[0].hist(y, bins=40, color='#3498db', edgecolor='white', alpha=0.85)
axes[0].axvline(y.mean(),   color='red',    linestyle='--', linewidth=2,
                label=f"Moyenne = {y.mean():.1f}%")
axes[0].axvline(y.median(), color='orange', linestyle='--', linewidth=2,
                label=f"Mediane = {y.median():.1f}%")
axes[0].set_xlabel("Ja in Prozent (%)")
axes[0].set_ylabel("Nombre de communes")
axes[0].set_title("Histogramme")
axes[0].legend()

# Les moustaches du boxplot correspondent exactement aux bornes IQR calculees
# ci-dessus. Les points hors moustaches sont les memes outliers (rouge Fig.1).
axes[1].boxplot(y, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.5),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.5, color='#e74c3c'))
axes[1].set_ylabel("Ja in Prozent (%)")
axes[1].set_title("Boxplot\n(points hors moustaches = outliers IQR)")
axes[1].set_xticks([])

plt.tight_layout()
plt.savefig("eda_fig_02_distribution.png", dpi=150)
plt.show()
print("  -> Figure 2 sauvegardee : eda_fig_02_distribution.png")


#
# 3.  VALEURS MANQUANTES APRÈS MERGE
#
section("3. VALEURS MANQUANTES APRES MERGE")

# Apres un LEFT JOIN, une commune sans correspondance dans un dataset secondaire
# aura des NaN pour TOUTES les colonnes de ce dataset.
miss_pct     = (train_m.isnull().mean() * 100).sort_values(ascending=False)
miss_nonzero = miss_pct[miss_pct > 0]

print(f"\n  Colonnes sans NaN   : {(miss_pct == 0).sum()} / {len(miss_pct)}")
print(f"  Colonnes avec NaN   : {len(miss_nonzero)}")
print(f"  Dont > 50% manquant : {(miss_nonzero > 50).sum()}")
print(f"  Dont > 20% manquant : {(miss_nonzero > 20).sum()}")
print(f"\n  Top 25 colonnes les plus incompletes :")
print(miss_nonzero.head(25).round(1).to_string())

# Communes non matchees par source
print("\n  Communes sans correspondance par source :")
for suffix, label in [('_622', '622'), ('_income', 'income')]:
    cols_src = [c for c in train_m.columns if c.endswith(suffix)]
    if cols_src:
        n = train_m[cols_src].isnull().all(axis=1).sum()
        print(f"    {label:10s} : {n} communes ({n/len(train_m)*100:.1f}%)")
geo_present = [c for c in df_geo.columns if c != 'Id' and c in train_m.columns]
if geo_present:
    n = train_m[geo_present].isnull().all(axis=1).sum()
    print(f"    {'geo':10s} : {n} communes ({n/len(train_m)*100:.1f}%)")

# Figure 3 : Barplot horizontal des NaN
if len(miss_nonzero) > 0:
    fig, ax = plt.subplots(figsize=(10, max(5, len(miss_nonzero) * 0.22)))
    colors_m = ['#e74c3c' if v > 50 else '#f39c12' if v > 20 else '#3498db'
                for v in miss_nonzero.values]
    ax.barh(miss_nonzero.index[::-1], miss_nonzero.values[::-1],
            color=colors_m[::-1], edgecolor='white')
    ax.axvline(50, color='red',    linestyle='--', linewidth=1, label="> 50%")
    ax.axvline(20, color='orange', linestyle='--', linewidth=1, label="> 20%")
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne (train fusionne)\nRouge > 50% | Orange > 20% | Bleu <= 20%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("eda_fig_03_missing.png", dpi=150)
    plt.show()


#
# 4.  CORRÉLATIONS SANS LEAKAGE
#
section("4. CORRELATIONS AVEC LA CIBLE")

# On retire les colonnes de leakage + la cible elle-meme,
# puis on ne garde que les colonnes numeriques.
cols_drop = [c for c in LEAKAGE_COLS + [TARGET] if c in train_m.columns]
X_num     = train_m.select_dtypes(include=np.number).drop(columns=cols_drop)
y_train   = train_m[TARGET]

print(f"\n  {X_num.shape[1]} features numeriques disponibles")

# Correlation de Pearson : mesure la relation LINEAIRE entre deux variables.
# r proche de +1 -> les deux variables augmentent ensemble.
# r proche de -1 -> quand l'une augmente, l'autre diminue.
# r proche de  0 -> pas de relation lineaire detectee.
# La p-value indique si la correlation est statistiquement fiable (p < 0.05).
corr_results = []
for col in X_num.columns:
    valid = X_num[col].notna() & y_train.notna()
    if valid.sum() < 10:
        continue
    r, p = pearsonr(X_num[col][valid], y_train[valid])
    corr_results.append({'feature': col, 'r': round(r, 4),
                         '|r|': round(abs(r), 4), 'p_value': round(p, 4),
                         'n_valid': valid.sum()})

corr_df = (pd.DataFrame(corr_results)
           .sort_values('|r|', ascending=False)
           .reset_index(drop=True))

sig_df    = corr_df[corr_df['p_value'] < 0.05]
strong_df = corr_df[corr_df['|r|'] >= 0.5]
mod_df    = corr_df[corr_df['|r|'] >= 0.3]

print(f"  Features significatives (p<0.05)   : {len(sig_df)} / {len(corr_df)}")
print(f"  Fortement correlees (|r| >= 0.5)   : {len(strong_df)}")
print(f"  Moderement correlees (|r| >= 0.3)  : {len(mod_df)}")
print(f"\n  --- Top 30 features correlees ---")
print(corr_df.head(30).to_string(index=False))

if len(strong_df):
    print(f"\n  --- Features FORTEMENT correlees (|r| >= 0.5) ---")
    print(strong_df[['feature', 'r', 'p_value']].to_string(index=False))

corr_df.to_csv("eda_table_correlations.csv", index=False)
print("\n  -> Tableau sauvegarde : eda_table_correlations.csv")

# Figure 4 : Barplot top 30 corrélations
top30    = corr_df.head(30).copy()
colors_c = ['#2ecc71' if r > 0 else '#e74c3c' for r in top30['r']]

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(top30['feature'][::-1], top30['r'][::-1],
        color=colors_c[::-1], edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
for i, (_, row) in enumerate(top30[::-1].iterrows()):
    offset = 0.005 if row['r'] >= 0 else -0.005
    ax.text(row['r'] + offset, i, f"{row['r']:.3f}",
            va='center', ha='left' if row['r'] >= 0 else 'right', fontsize=7)
ax.set_xlabel("Correlation de Pearson r")
ax.set_title("Top 30 features correlees avec 'Ja in Prozent'\n(vert = positif, rouge = negatif | sans leakage)")
plt.tight_layout()
plt.savefig("eda_fig_04_correlations.png", dpi=150)
plt.show()
print("\n  -> Figure 4 sauvegardee : eda_fig_04_correlations.png")

# Figure 5 : Scatter plots top 6 features
top6 = corr_df.head(6)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Top 6 features vs. 'Ja in Prozent' (sans leakage)", fontsize=13, fontweight='bold')

for ax, col in zip(axes.flat, top6):
    x_col = train_m[col]
    valid  = x_col.notna() & y_train.notna()
    r_val  = corr_df.loc[corr_df['feature'] == col, 'r'].values[0]

    ax.scatter(x_col[valid], y_train[valid], alpha=0.3, s=8, color='#3498db')
    # Droite de regression (tendance lineaire)
    z = np.polyfit(x_col[valid], y_train[valid], 1)
    xline = np.linspace(x_col[valid].min(), x_col[valid].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color='red', linewidth=1.5)

    ax.set_xlabel(col[:35], fontsize=8)
    ax.set_ylabel("Ja in Prozent", fontsize=8)
    ax.set_title(f"r = {r_val:.3f}", fontsize=10)

plt.tight_layout()
plt.savefig("eda_fig_05_scatter_top6.png", dpi=150)
plt.show()
print("  -> Figure 5 sauvegardee : eda_fig_05_scatter_top6.png")


#
# 5.  PATTERNS & ANOMALIES
#
section("5. PATTERNS & ANOMALIES")

#5a. Analyse par canton
canton_col ="Kanton"

print(f"\n  [5a] Analyse par canton (colonne : '{canton_col}')")
stats_c = (train_m.groupby(canton_col)[TARGET]
               .agg(Moyenne='mean', Std='std', N='count', Min='min', Max='max')
               .round(2).sort_values('Moyenne', ascending=False))
print(stats_c.to_string())
print(f"\n  -> Canton le plus OUI  : {stats_c.index[0]}  ({stats_c['Moyenne'].iloc[0]:.1f}%)")
print(f"  -> Canton le moins OUI : {stats_c.index[-1]}  ({stats_c['Moyenne'].iloc[-1]:.1f}%)")
print(f"  -> Ecart entre cantons : {stats_c['Moyenne'].max() - stats_c['Moyenne'].min():.1f} points")

fig, ax = plt.subplots(figsize=(14, 5))
order = stats_c.index.tolist()
sns.boxplot(data=train_m, x=canton_col, y=TARGET, order=order, palette='viridis', ax=ax)
ax.set_xlabel("Canton")
ax.set_ylabel("Ja in Prozent (%)")
ax.set_title("Distribution du % OUI par canton (tries par moyenne decroissante)")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("eda_fig_06_canton.png", dpi=150)
plt.show()
print("\n  -> Figure 6 sauvegardee : eda_fig_06_canton.png")


# ── 5b. Multicolinéarité entre les top features ──────────────────────────────
# Si deux features sont tres correlees entre elles (|r| > 0.8), elles apportent
# la meme information. En garder les deux n'aide pas le modele et peut le
# perturber. On les detecte pour decider laquelle garder.
print(f"\n  [5b] Multicolinearite – paires de features avec |r| > 0.8")

#on verifie que les variables les plus corrélées avec le target ne se repettent pas
top15_feat = [f for f in corr_df.head(15)['feature'] if f in train_m.columns]
#matrice 15x15 de correlations
corr_matrix = train_m[top15_feat].corr()

pairs_found = False

cols_ = corr_matrix.columns.tolist()

"""
On fais une double boucle afin de comparé entre elle les features du top 15 de celles les plus 
corrélée avec la target.
On verifie A <-> B mais on evite B <-> car la correlation serait la même
On ne verifie pas A <-> A car ça vaut toujours 1 
"""
for i in range(len(cols_)):
    for j in range(i + 1, len(cols_)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.8:
            print(f"    {cols_[i][:35]} <-> {cols_[j][:35]:35s}  r = {c:.3f}")
            pairs_found = True

