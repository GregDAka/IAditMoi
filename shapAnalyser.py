import shap
import numpy as np
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
        self.shap_values = self.explainer(self.data)
        print(np.shape(self.shap_values))

    def plot_waterfall(self, classes, prediction):
        """
            index: Index de l'observation à analyser.
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        #shap.waterfall_plot(self.shap_values[index])
        #si classes du dataset ne sont pas bianire (ex:van, saab, opel, bus)
        if classes.__len__() > 2:
            #on recherche l'indice correspondant à la prédiction pour donner la bonne dimension de shap values
            for i in range(classes.__len__()):
                if classes[i] == prediction:
                    raw_prediction = i
            fig, ax = plt.subplots()
            shap.plots.waterfall(self.shap_values[0][:,raw_prediction], show=False)
            st.pyplot(fig)
        #sinon
        else:

            fig, ax = plt.subplots()
            shap.plots.waterfall(self.shap_values[0], show=False)
            st.pyplot(fig)

    
    def Anal_summary_plot(self, classes, prediction):
        #si classes du dataset ne sont pas bianire (ex:van, saab, opel, bus)
        if classes.__len__() > 2:
            #on recherche l'indice correspondant à la prédiction pour donner la bonne dimension de shap values
            for i in range(classes.__len__()):
                if classes[i] == prediction:
                    raw_prediction = i
            fig, ax = plt.subplots()
            shap.summary_plot(self.shap_values[:,:,raw_prediction], self.data)
            st.pyplot(fig)
        #sinon
        else:
            fig, ax = plt.subplots()
            shap.summary_plot(self.shap_values, self.data)
            st.pyplot(fig)
        

    def Anal_heatmap(self, classes):
        if classes.__len__() <= 2:
            fig, ax = plt.subplots()
            shap.plots.heatmap(self.shap_values)
            st.pyplot(fig)

    