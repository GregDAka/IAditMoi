from sklearn.svm import LinearSVC

class LinearSVCModel():

    def __init__(self):
        self.model = LinearSVC()


    def predict(self,X, Y, features):
        self.model.fit(X, Y)
        print(self.model.score(X, Y))
        print("J'utilise un LinearSVC")
        return self.model.predict(features)
