import pandas as pd
from sklearn import datasets
import streamlit as st
from Model import Model
from LinearSVCModel import LinearSVCModel
from SGDclassifierModel import SGDclassifierModel
from Data import Data



vehicule = Data("vehicule.csv")
print(vehicule.dfY.head())
lsvc = LinearSVCModel()
model = Model(lsvc, vehicule, vehicule.dfX[0:1:])
prediction = model.predict()
print(prediction)