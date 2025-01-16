from Data import Data

class Model():

    def __init__(self, concrete_model, data : Data, user_inputs):
        self.setModel(concrete_model)
        self.X = data.dfX
        self.Y = data.dfY
        self.user_inputs = user_inputs

    def setModel(self, concrete_model):
            self.model_chosed = concrete_model

    def predict(self):
        return self.model_chosed.predict(self.X, self.Y, self.user_inputs)
    
    def getConcreteModel(self):
         return self.model_chosed.getConcreteModel(self.X, self.Y)





    