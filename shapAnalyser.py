import shap
import matplotlib.pyplot as plt

class SHAPAnalyzer:
    def __init__(self, model, data):
        """
        model: Le modèle entraîné (LinearSVC(concreteModel dans le main))
        data: Les données utilisées pour expliquer les prédictions. (data.dfX dans le main)
        """
        self.model = model
        self.data = data
        self.explainer = shap.Explainer(model, data)
        self.shap_values = None
    
    
    def compute_shap_values(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values = self.explainer(self.data)

    def plot_waterfall(self, index):
        """
            index: Index de l'observation à analyser.
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        shap.waterfall_plot(self.shap_values[index])