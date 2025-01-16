import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances

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
        self.shap_values_ui = None
    
    
    def compute_shap_values(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values = self.explainer(self.data)
        print(np.shape(self.shap_values))

    def compute_shap_values_ui(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values_ui = self.explainer(self.user_inputs_df)
        print(np.shape(self.shap_values_ui))

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
            shap.plots.waterfall(self.shap_values_ui[0][:,raw_prediction], show=False)
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


    def compute_fidelity(self):
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")

        # Calcul des prédictions approximées par SHAP
        approx_preds = self.shap_values.values.sum(axis=1) + self.shap_values.base_values

        # Prédictions originales du modèle
        original_preds = self.model.predict(self.data)

        # Vérifiez si les prédictions sont des chaînes et encodez-les
        if isinstance(original_preds[0], str):
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            original_preds = label_encoder.fit_transform(original_preds)

        # Calcul de la fidélité
        fidelity = np.corrcoef(original_preds, approx_preds)[0, 1]

        return fidelity


    def compute_stability(self, num_samples=10, noise=0.01):
        """Calcule la stabilité des explications."""
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        stability_scores = []
        for _ in range(num_samples):
            perturbed_data = self.data + np.random.normal(0, noise, self.data.shape)
            perturbed_shap_values = self.explainer(perturbed_data)
            dist_original = euclidean_distances(self.shap_values.values, self.shap_values.values)
            dist_perturbed = euclidean_distances(perturbed_shap_values.values, perturbed_shap_values.values)
            stability_scores.append(np.corrcoef(dist_original.flatten(), dist_perturbed.flatten())[0, 1])
        return np.mean(stability_scores)


    def compute_robustness(self, perturbation_factor=0.1):
        """Calcule la robustesse des explications."""
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        perturbed_data = self.data + np.random.normal(0, perturbation_factor, self.data.shape)
        perturbed_shap_values = self.explainer(perturbed_data)
        robustness_score = np.linalg.norm(self.shap_values.values - perturbed_shap_values.values)
        return robustness_score

    