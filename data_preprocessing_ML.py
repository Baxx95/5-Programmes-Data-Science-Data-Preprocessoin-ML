# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:07:12 2021

@author: Zakaria
"""

#import statsmodels as stat
#import seaborn as sns
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import SimpleImputer
from sklearn.impute import SimpleImputer as SImpt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection  import train_test_split


dataset = pd.read_csv("credit_immo.csv")

print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset.columns)

X = dataset.iloc[:, -9:-1].values
Y = dataset.iloc[:,-1].values

print(dataset.isnull().sum())
print(dataset.isnull().sum)

#imptr = SImpt(missing_values="NAN", strategy='mean')
imptr = SImpt(missing_values=np.nan, strategy='mean')

X[:, 0:1]

imptr.fit(X[:, 0:1])
imptr.fit(X[:, 7:8])

# Imputer toutes les valeurs manquantes par une stratégie
X[:, 0:1] = imptr.transform(X[:, 0:1])
X[:, 7:8] = imptr.transform(X[:, 7:8])
X[:, 0:1]
X[:, 7:8]


#----------------------------------------------------#
############### DONNEES CATEGORIELLES ################
#----------------------------------------------------#
LabEncdr_X = LabelEncoder()

X[:,2] = LabEncdr_X.fit_transform(X[:,2])
X[:,5] = LabEncdr_X.fit_transform(X[:,5])

#OnehotEncr = OneHotEncoder(categorical_features=[2]) ---> la methode ne marche pas le param categorical_features
OnehotEncr = OneHotEncoder()
#OnehotEncr = OneHotEncoder(categorical_features=[5])

X = OnehotEncr.fit_transform(X).toarray()

#On réalise la même transformation sur Y : transformer les var_Catégorielles "OUI" et "NON" en "0" et "1"
Y = LabelEncoder().fit_transform(Y)


#---------------------------------------------------------#
##### FRACTIONNEMENT DATASET EN TRAIN_SET ET TEST8SET #####
#---------------------------------------------------------#

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#standardisation (Centrer-reduire) signifie conversion vers un standard commun
StdSc = StandardScaler()

X_train = StdSc.fit_transform(X_train)
X_test = StdSc.fit_transform(X_test)



X_train = normalize(X_train)
X_test = normalize(X_test)
