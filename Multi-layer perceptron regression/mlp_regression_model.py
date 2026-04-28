from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

train_df = pd.read_csv('../Data-Sets/results_train.csv') #données d'entrainements
test_df = pd.read_csv('../Data-Sets/results_test.csv') #données de tests
geo = pd.read_csv('../Data-Sets/swiss_communes_geodata.csv') #latitude and longitude of each commune.
demo = pd.read_excel("../Data-Sets/je-e-21.03.01.xlsx", header=5) #demographic, geographic, economic, and voting data about each Swisscommune in 2018
demo = demo.drop(0) # la ligne 7 ne contient que des dates
income = pd.read_excel("../Data-Sets/statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx") #income data for each Swiss com-mune in 2017.
prev_ref = pd.read_excel("../Data-Sets/622.00-result-by-canton-district-and-municipality.xlsx") #results of a previous ref-erendum “Initiative for food sovereignty”

#------------ On prepare les cols Id = numero de comunes
train_df['Id'] = train_df['Gemeinde-Nummer'].astype(str)
test_df['Id'] = test_df['Gemeinde-Nummer'].astype(str)

TARGET = 'Ja in Prozent'

#On enleve ces collones car elles correspondent au details des votes, on pourrait deduire la target juste avec des calculs
Leakage_cols = ['eingelegte Stimmzettel','Stimmbeteiligung','leere Stimmzettel', 'ungültige Stimmzettel', 'gültige Stimmen', 'Ja-Stimmen', 'Nein-Stimmen']

Y_train = train_df[TARGET].copy()
train_df = train_df.drop(columns = Leakage_cols + [TARGET])

#On supprimer les communes et les kanton car on a deja leurs numéro
#Peut etre faire un one hot encoder pour eviter que commune 1 < commune 2 ?

train = train.drop(columns = ['Kanton', 'Gemeinde'])
test = test.drop(columns = ['Kanton', 'Gemeinde'])

#on supprimer les noms de commune et on unifie les noms entre geo et train
geo = geo.drop(columns = 'municipalityLabel')
geo = geo.rename(columns = {'bfs_id':'Gemeinde-Nummer'})
#on fusionne geo et train (inner pour ne pas garder les nan):
train = train.merge(geo, on = 'Gemeinde-Nummer', how = 'left')
test = test.merge(geo, on = 'Gemeinde-Nummer', how = 'left')

#traitement demo :
demo = demo.rename(columns={'Number of commune' :'Gemeinde-Nummer' })
demo = demo.drop(columns = ['Name of commune'])
pd.set_option('display.max_columns', None)
#print(train.info())
#print(test.info())
print(demo.info())
print(demo.head)










