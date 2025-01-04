import shap
import matplotlib.pyplot as plt
import streamlit as st

class SHAPAnalyzer:
    def __init__(self, model, data, user_inputs_df):
        """
        model: Le modèle entraîné (LinearSVC(concreteModel dans le main))
        data: Les données utilisées pour expliquer les prédictions. (data.dfX dans le main)
        """
        self.model = model
        self.data = data
        self.explainer = shap.Explainer(model, data)
        self.user_inputs_df = user_inputs_df
        self.shap_values = None
    
    
    def compute_shap_values(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values = self.explainer(self.user_inputs_df)

    def plot_waterfall(self,shap_values, classes, prediction):
        """
            index: Index de l'observation à analyser.
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        #shap.waterfall_plot(self.shap_values[index])
        for i in range(classes.__len__()):
            if classes[i] == prediction:
                raw_prediction = i
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0][:,raw_prediction], show=False)
        st.pyplot(fig)