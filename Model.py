class Model():

    def __init__(self, concrete_model , X, Y, features):
        self.setModel(concrete_model)
        self.X = X
        self.Y = Y
        self.features = features

    def setModel(self, concrete_model):
            self.model_chosed = concrete_model
        

    def predict(self):
        return self.model_chosed.predict(self.X, self.Y, self.features)
       




    