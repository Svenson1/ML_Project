from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

train = pd.read_csv('../Data-Sets/results_train.csv') #données d'entrainements
test = pd.read_csv('../Data-Sets/results_test.csv') #données de tests
geo = pd.read_csv('../Data-Sets/swiss_communes_geodata.csv') #latitude and longitude of each commune.
demo = pd.read_excel("je-e-21.03.01.xlsx") #demographic, geographic, economic, and voting data about each Swisscommune in 2018
income = pd.read_excel("statistik-dbst-np-kennzahlen-mit-2017-fr.xlsx") #income data for each Swiss com-mune in 2017.
prev_ref = pd.read_excel("622.00-result-by-canton-district-and-municipality.xlsx") #results of a previous ref-erendum “Initiative for food sovereignty”






