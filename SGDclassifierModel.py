from sklearn.linear_model import SGDClassifier

"""
Cette classes est un des implémentation concrete de la prédiction pour 
le modèle SGDClassifier.
Attibut:
    model: SGDClassifier (objet scikit-learn)

"""
class SGDclassifierModel():

    def __init__(self):
        self.model = SGDClassifier()

    def predict(self,X, Y, features):
        """
        Fournie les data nécessaire poue entrainer le modèle puis prédire.
        Renvoie la prédiction et le score du model.
        X: les attributs 
        Y: les classes
        user_inputs: les entrée de l'utilisateur (sliders)
        """
        self.model.fit(X, Y)
        accuracy = self.model.score(X, Y)
        print("J'utilise un SGDClassifier")
        prediction = self.model.predict(features)
        return prediction, accuracy
    
    def getConcreteModel(self,X,Y):
         """
         Renvoie le modèle concret (objet scikit-learn)  avec les données.
         """
         self.model.fit(X, Y)
         return self.model