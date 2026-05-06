import pandas as pd


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

def import_datasets():
    # training and test set :
    train_df = pd.read_csv("../Data_Sets/results_train.csv")
    test_df = pd.read_csv("../Data_Sets/results_test.csv")
    train_df['Id'] = train_df['Gemeinde-Nummer'].astype(
        str)  # On ajoute une colone id qui est egale au numero de commune
    test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)
    train_df = train_df.drop(columns=['Gemeinde-Nummer'])
    test_df = test_df.drop(columns=['Gemeinde-Nummer'])

    # Other referundum = 622
    file_622 = "../Data_Sets/622.00-result-by-canton-district-and-municipality.xlsx"
    df_622 = pd.read_excel(file_622, sheet_name="Gemeinden", header=5)
    df_622.columns = df_622.columns.str.strip()
    df_622 = clean_and_format_id(df_622, 'Gemeinde-Nummer')
    df_622 = df_622.drop_duplicates(subset=['Id'])
    df_622 = df_622.add_suffix('_622')
    df_622 = df_622.rename(columns={'Id_622': 'Id'})  # pour la fusion apres
    df_622 = df_622.drop(columns=['Gemeinde-Nummer_622', 'Gemeinde_622', 'Kanton_622'])

    # portrait of communes = jee
    file_jee = "../Data_Sets/je-e-21.03.01.xlsx"
    df_jee = pd.read_excel(file_jee, sheet_name="Schweiz - Gemeinden", header=5)
    df_jee = clean_and_format_id(df_jee, 'Number of commune')
    df_jee = df_jee.drop_duplicates(subset=['Id'])
    df_jee = df_jee.drop(columns=['Number of commune', 'Name of commune'])
    # On force les cols a être des chiffres :
    for col in df_jee.columns:
        if col != 'Id':
            df_jee[col] = pd.to_numeric(df_jee[col], errors='coerce')

    # geoData
    file_geo = "../Data_Sets/swiss_communes_geodata.csv"
    df_geo = pd.read_csv(file_geo)
    df_geo = clean_and_format_id(df_geo, 'bfs_id')
    df_geo = df_geo.drop_duplicates(subset=['Id'])
    df_geo = df_geo.drop(columns=['bfs_id', 'municipalityLabel'])

    # income data for each Swiss com-mune in 2017
    file_income = "../Data_Sets/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx"
    df_income = pd.read_excel(file_income, sheet_name='Gemeinden - Communes')
    df_income = clean_and_format_id(df_income, 'gdenr')
    df_income = df_income.drop_duplicates(subset=['Id'])
    df_income = df_income.drop(columns=['ktname', 'gdename', 'Einheit'])
    df_income = df_income.add_suffix('_income')
    df_income = df_income.rename(columns={'Id_income': 'Id'})
    # On force les cols a être des chiffres :
    for col in df_income.columns:
        if col != 'Id':
            df_income[col] = pd.to_numeric(df_income[col], errors='coerce')

    return train_df, test_df, df_622, df_jee, df_geo, df_income


def import_and_merge_dataset():
    train_df, test_df, df_622, df_jee, df_geo, df_income = import_datasets()
    train_merged = (train_df.merge(df_622, on='Id', how='left')
                    .merge(df_jee, on='Id', how='left')
                    .merge(df_income, on='Id', how='left')
                    .merge(df_geo, on='Id', how='left'))

    test_merged = (test_df.merge(df_622, on='Id', how='left')
                   .merge(df_jee, on='Id', how='left')
                   .merge(df_income, on='Id', how='left')
                   .merge(df_geo, on='Id', how='left'))

    return train_merged, test_merged