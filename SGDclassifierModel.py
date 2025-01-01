from sklearn.linear_model import SGDClassifier

class SGDclassifierModel():

    def __init__(self):
        self.model = SGDClassifier()

    def predict(self,X, Y, features):
        self.model.fit(X, Y)
        print(self.model.score(X, Y))
        print("J'utilise un SGDClassifier")
        return self.model.predict(features)
    
    def getConcreteModel(self,X,Y):
         self.model.fit(X, Y)
         return self.model