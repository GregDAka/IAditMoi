import shap
from sklearn.linear_model import SGDClassifier


X, y = shap.datasets.adult()
model = SGDClassifier()
model.fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap.plots.waterfall(shap_values[0])
