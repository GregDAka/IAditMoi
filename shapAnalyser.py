import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder

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
            label_encoder = LabelEncoder()
            original_preds = label_encoder.fit_transform(original_preds)

        # Convertir les prédictions en tableaux NumPy
        original_preds = np.array(original_preds)
        approx_preds = np.array(approx_preds)

        # Si original_preds est 1D, redimensionnez en (n_samples, 1)
        if len(original_preds.shape) == 1:
            original_preds = original_preds.reshape(-1, 1)

        # Si approx_preds est 1D (cas rare), redimensionnez en (n_samples, 1)
        if len(approx_preds.shape) == 1:
            approx_preds = approx_preds.reshape(-1, 1)

        # Ajuster les dimensions si nécessaire
        if original_preds.shape[1] != approx_preds.shape[1]:
            if original_preds.shape[1] == 1:  # Une seule cible dans original_preds
                original_preds = np.tile(original_preds, (1, approx_preds.shape[1]))
            elif approx_preds.shape[1] == 1:  # Une seule cible dans approx_preds
                approx_preds = np.tile(approx_preds, (1, original_preds.shape[1]))
            else:
                raise ValueError(
                    f"Dimension mismatch: original_preds={original_preds.shape}, approx_preds={approx_preds.shape}"
                )

        # Calcul de la fidélité pour chaque cible
        fidelity_scores = []
        for target_idx in range(original_preds.shape[1]):
            fidelity = np.corrcoef(original_preds[:, target_idx], approx_preds[:, target_idx])[0, 1]
            fidelity_scores.append(fidelity)

        # Retourner la moyenne des scores si plusieurs cibles, ou le score unique pour une seule cible
        return np.mean(fidelity_scores) if len(fidelity_scores) > 1 else fidelity_scores[0]

       
    def compute_stability(self, num_samples=10, noise=0.2, aggregation_method='mean'):
        """
        Calcule la stabilité des explications.
        """
        if self.shap_values is None:
            raise ValueError("Les valeurs SHAP n'ont pas encore été calculées.")

        stability_scores = []

        # Fonction pour transformer les valeurs SHAP en 2D
        

        # Appliquer l'agrégation sur les valeurs originales
        original_shap_2d = self.process_shap_values(self.shap_values.values, aggregation_method)

        for _ in range(num_samples):
            # Ajouter du bruit pour perturber les données
            perturbed_data = self.data + np.random.normal(0, noise, self.data.shape)

            # Calculer les nouvelles valeurs SHAP
            perturbed_shap_values = self.explainer(perturbed_data)
            perturbed_shap_2d = self.process_shap_values(perturbed_shap_values.values, aggregation_method)

            # Calculer les distances euclidiennes
            dist_original = euclidean_distances(original_shap_2d, original_shap_2d)
            dist_perturbed = euclidean_distances(perturbed_shap_2d, perturbed_shap_2d)

            # Corrélation entre les distances
            stability_scores.append(np.corrcoef(dist_original.flatten(), dist_perturbed.flatten())[0, 1])

        return np.mean(stability_scores)
    

    def process_shap_values(self, shap_values, method):
        if len(shap_values.shape) == 3:  # SHAP values avec plusieurs targets
            if method == 'mean':
                return shap_values.mean(axis=-1)  # Moyenne sur la dernière dimension
            elif method == 'sum':
                return shap_values.sum(axis=-1)  # Somme sur la dernière dimension
            elif method == 'flatten':
                return shap_values.reshape(shap_values.shape[0], -1)  # Aplatir complètement
            else:
                raise ValueError(f"Méthode d'agrégation inconnue : {method}")
        return shap_values  # Si déjà 2D, on retourne tel quel


    