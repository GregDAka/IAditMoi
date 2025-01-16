from sklearn.svm import LinearSVC

class LinearSVCModel():

    def __init__(self):
        self.model = LinearSVC()

    def predict(self,X, Y, user_inputs):
        self.model.fit(X, Y)
        accuracy = self.model.score(X, Y)
        print("J'utilise un LinearSVC")
        prediction = self.model.predict(user_inputs)
        return prediction, accuracy

    def getConcreteModel(self,X,Y):
         self.model.fit(X, Y)
         return self.model