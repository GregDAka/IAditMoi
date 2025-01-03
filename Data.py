import pandas as pd
import csv 

class Data : 

    def __init__(self,csv_file:str):
        self.dfX,self.dfY=self.createData(csv_file)


    def createData(self, csv_file):
        dataframe=pd.read_csv(csv_file) # Notre fichier contient une ligne d'entête qu'on ne veut pas garder, donc on utilise pas header=0
        dfX=dataframe.iloc[:,:-1] #Toutes les colones sauf la dernière
        dfY=dataframe.iloc[:,-1:] #Que la dernière colone
        return dfX,dfY

    def collectDataAttributes(self):
        return list(self.dfX.columns)  # On récupère le nom de chaque colonne sauf la dernière

    def collectDataClasses(self):
        return self.dfY.iloc[:, 0].unique()  # On récupère sans répétition le contenu de la dernière colonne 

    def collectMinMaxValues(self, attribute):
        return self.dfX[attribute].min(), self.dfX[attribute].max() # On récupère la valeur minimum et maximum d'une colonne donnée (sauf la dernière)'
