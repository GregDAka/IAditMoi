import pandas as pd

"""
Cette classe créer un ojbet Data contenant les deux parties du tableau fournis.
Attributs:
    dfX les attributs
    dfY les classes
"""
class Data : 

    def __init__(self,csv_file:str):
        """
        Fais le séparation du tableau pour initialiser les attributs
        """
        dataframe=pd.read_csv(csv_file) # Notre fichier contient une ligne d'entête qu'on ne veut pas garder, donc on utilise pas header=0
        self.dfX=dataframe.iloc[:,:-1] #Toutes les colones sauf la dernière
        self.dfY=dataframe.iloc[:,-1:] #Que la dernière colone

    def collectDataAttributes(self):
        """
        Récupère le nom de chaque colonne sauf la dernière
        """
        return list(self.dfX.columns)  

    def collectDataClasses(self):
        """
        Récupère sans répétition le contenu de la dernière colonne (les classes)
        """
        return self.dfY.iloc[:, 0].unique() 
    
    def collectMinMaxValues(self, attribute):
        """
        Récupère la valeur minimum et maximum d'une colonne donnée (sauf la dernière)
        attribute: l'attribut dont on veux récupérer les valeure max et min
        """
        return self.dfX[attribute].min(), self.dfX[attribute].max()  
