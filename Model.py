from Data import Data

"""
Cette classe a pour but de nous permtre de choisir le véritable modèle que l'on veut utiliser
et le manipuler à travers cette classe.
Attributs :
    X: Données des attributs dans le csv.
    Y: Les classes issuent du csv.
    user_inputs: Données fournient par l'utlisateur au travers des sliders.
"""
class Model():

    def __init__(self, concrete_model, data : Data, user_inputs):
        self.setModel(concrete_model)
        self.X = data.dfX
        self.Y = data.dfY
        self.user_inputs = user_inputs

    def setModel(self, concrete_model):
            """
            Change le modèle concret à utliser.
            concrete_model: classe concrete à set
            """
            self.model_chosed = concrete_model

    def predict(self):
        """
        Appel le prédiction de véritable model puis la renvoie.
        """
        return self.model_chosed.predict(self.X, self.Y, self.user_inputs)
    
    def getConcreteModel(self):
         """
         Demande eu modèle concret de renvoyer le model (objet scikit-learn) puis le revoie.
         """
         return self.model_chosed.getConcreteModel(self.X, self.Y)





    