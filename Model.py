class Model():

    def __init__(self, concrete_model , data, features):
        self.setModel(concrete_model)
        self.X = data.dfX
        self.Y = data.dfY
        self.features = features

    def setModel(self, concrete_model):
            self.model_chosed = concrete_model
        

    def predict(self):
        return self.model_chosed.predict(self.X, self.Y, self.features)
       




    