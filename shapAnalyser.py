import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder

class SHAPAnalyzer:
    def __init__(self, model, data, user_inputs_df):
        """
        model: Le modèle entraîné (par exemple, LinearSVC).
        data: Données sur lesquelles le modèle a été entraîné (ex. data.dfX).
        user_inputs_df: Données fournies par l'utilisateur pour une analyse personnalisée.
        """
        self.model = model
        self.data = data
        self.explainer = shap.Explainer(model, data)  # Initialisation de l'explainer SHAP.
        self.user_inputs_df = user_inputs_df  # Données utilisateur pour l'analyse.
        self.shap_values = None  # Valeurs SHAP calculées pour les données globales.
        self.shap_values_ui = None  # Valeurs SHAP calculées pour les données utilisateur.


    def compute_shap_values(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values = self.explainer(self.data)  # Calcule les valeurs SHAP pour les données globales.
        print(np.shape(self.shap_values))  # Affiche les dimensions des valeurs SHAP.

    def compute_shap_values_ui(self):
        """Calcule les valeurs SHAP pour les données fournies."""
        self.shap_values_ui = self.explainer(self.user_inputs_df)  # Calcule les valeurs SHAP pour une entrée utilisateur.
        print(np.shape(self.shap_values_ui))  # Affiche les dimensions des valeurs SHAP pour cette entrée.

    def plot_waterfall(self, classes, prediction):
        """
        Génère un waterfall plot pour une observation spécifique.
        classes: Liste des classes possibles dans le JDD
        prediction: Classe prédite pour l'observation.
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")
        if classes.__len__() > 2:
            #si classes du dataset ne sont pas bianire (ex:van, saab, opel, bus)
            # Trouve l'indice correspondant à la classe prédite
            for i in range(classes.__len__()):
                if classes[i] == prediction:
                    raw_prediction = i
            fig, ax = plt.subplots()
            shap.plots.waterfall(self.shap_values_ui[0][:,raw_prediction], show=False) # Affiche les valeurs SHAP pour la classe cible.
            st.pyplot(fig)
        else:
            #cas binaire
            fig, ax = plt.subplots()
            shap.plots.waterfall(self.shap_values[0], show=False)
            st.pyplot(fig)

    
    def Anal_summary_plot(self, classes, prediction):
        """
        Génère un summary plot pour visualiser les tendances globales.
        classes: Liste des classes possibles dans le JDD
        prediction: Classe prédite pour l'observation.
        """
        if classes.__len__() > 2:
            #on recherche l'indice correspondant à la prédiction pour donner la bonne dimension de shap values
            for i in range(classes.__len__()):
                if classes[i] == prediction:
                    raw_prediction = i
            fig, ax = plt.subplots()
            shap.summary_plot(self.shap_values[:,:,raw_prediction], self.data)# Visualisation multiclasse.
            st.pyplot(fig)
        #sinon
        else:
            fig, ax = plt.subplots()
            shap.summary_plot(self.shap_values, self.data) # Visualisation binaire.
            st.pyplot(fig)
        
    
    def Anal_heatmap(self, classes):
        """
        Génère un heatmap plot pour visualiser les interactions entre caractéristiques.
        classes: Liste des classes possibles dans le JDD.
        """
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
    

    def compute_stability(self, num_samples=10, noise=0.2):
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

    