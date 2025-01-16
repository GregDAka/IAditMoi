from sklearn.svm import LinearSVC

"""
Cette classes est un des implémentation concrete de la prédiction pour 
le modèle LinearSVC.
Attibut:
    model: LinearSVC (objet scikit-learn)

"""
class LinearSVCModel():

    def __init__(self):
        self.model = LinearSVC()

    def predict(self,X, Y, user_inputs):
        """
        Fournie les data nécessaire poue entrainer le modèle puis prédire.
        Renvoie la prédiction et le score du model.
        X: les attributs 
        Y: les classes
        user_inputs: les entrée de l'utilisateur (sliders)
        """
        self.model.fit(X, Y)
        accuracy = self.model.score(X, Y)
        print("J'utilise un LinearSVC")
        prediction = self.model.predict(user_inputs)
        return prediction, accuracy

    def getConcreteModel(self,X,Y):
        """
        Renvoie le modèle concret (objet scikit-learn)  avec les données.
        """
        self.model.fit(X, Y)
        return self.model