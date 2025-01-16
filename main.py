import pandas as pd
import streamlit as st
from Model import Model
from LinearSVCModel import LinearSVCModel
from SGDclassifierModel import SGDclassifierModel
from shapAnalyser import SHAPAnalyzer
from Data import Data

st.set_page_config(page_title="Model Explainability")
# Ici on gère l'import du CSV

st.title("Machine Learning Model Explanation Project")
st.write('''
To begin, import your data in csv form. The first line of the csv are the attributes and the last column is the target
''')
upload = st.file_uploader("Choose your data", type="csv")

if upload is not None:
    data = Data(upload)

    attributes = data.collectDataAttributes() # On extrait les noms différents attributs du JDD (ex : *taille* de x, *longueur* de x etc)
    classes = data.collectDataClasses() # On extrait les différentes classes du JDD (ex : chat, chien, vache)

    st.write("Welcome, this application simplifies ML usage and understanding by providing through SHAP Explanations and Quality Metrics for predictions.")
    st.write("This project was developped by four students - DUBOULOY Grégory, DUFOUR Benoît, DELEMASURE Gabriel and BEAUCOUSIN Enzo as part of an academic project. We hope that through its usage, this application proves useful to you.")

    # On définie de manière modulaire nos sliders de customisation des paramètres d'attributs à gauche dans l'interface
    st.sidebar.header("Slide to Modify Attribute Values")
    user_inputs = {} # Dictionnaire pour stocker les valeurs définies par l'utilisateur via les sliders pour chaque attribut
    for attribute in attributes: # Boucle qui va créer un slider de customisation pour chaque attribut précédemment extraits présents dans le jeu de donnée
        min_val, max_val = data.collectMinMaxValues(attribute)
        user_inputs[attribute] = st.sidebar.slider( 
            f"{attribute}", # Pour afficher le nom de la colonne/le nom de l'attribut correspondant au dessus du slider créé
            min_value=float(min_val), # Défini le min du slider selon la plus petite valeur contenue dans la colonne de l'attribut
            max_value=float(max_val), # Défini le max du slider selon la plus grande valeur contenue dans la colonne de l'attribut
            value=float((min_val + max_val) / 2),  # Pour placer le slider au milieu par défaut /Pour définir par défaut la valeur à la moitié du champ possible
        )

    st.subheader("Attribute Values Being Subject to the Prediction")
    st.write("(Slide to consult undisplayed attributes if needed)")
    st.table([user_inputs]) # On affiche le tableau qui contient tout les attributs et toutes les valeurs associés qu'on a définié via les sliders

    st.subheader("Potential Prediction Results: ")
    st.table([{"Class": cls} for cls in classes]) # On affiche dans un tableau toutes les Class potentielles à la prédiction qui existent dans le fichier csv

    user_inputs_df = pd.DataFrame([user_inputs]) # On converti les entrées utilisateurs(valeurs des sliders) en DataFrame pour la prédiction

    # On crée un dictionnaire avec la liste des modèles en place
    modelOptions = {
        "LinearSVC": LinearSVCModel,
        "SGDclassifier": SGDclassifierModel,
    }

    #On crée une boîte de sélection dans l'interface pour rendre le choix du modèle flexible 
    #(Permet aussi de changer de modèle sans avoir à redémarrer l'appli, bien plus pratique à l'utilisation)
    modelUserSelection = st.selectbox("Select a Model (LinearSVC is used by default)", list(modelOptions.keys()))
        
    #On met en place le modèle choisi et sa prédiction puis l'effectue
    chosenModel = modelOptions[modelUserSelection]
    model = Model(chosenModel(), data, user_inputs_df)
    prediction, accuracy = model.predict()
    concreteModel = model.getConcreteModel()

    st.write(f"Model accuracy on chosen dataset : {accuracy:.4f}")
    st.write("The predicted class is :",prediction) # On affiche la prédiction

    colExp,colMet = st.columns([70,30],vertical_alignment="center")
    col1,col2 = st.columns([70,30],vertical_alignment="top")

    with colExp:
        st.subheader("SHAP Explanation Plots")
    with colMet:
        st.subheader("Quality Metrics")

    with col1:
        # On calcule les valeurs SHAP pour les valeurs fournies
        shap_analyzer = SHAPAnalyzer(concreteModel,data.dfX, user_inputs_df)
        shap_analyzer.compute_shap_values()
        shap_analyzer.compute_shap_values_ui()

        # On affiche les plots SHAP 
        st.write("Waterfall Plot: useful for analyzing feature contributions for an individual observation")
        with st.spinner("Loading the Waterfall"):
            shap_analyzer.plot_waterfall(classes, prediction)

        st.write("Summary Plot: provides a comprehensive overview of important features across all observations")
        with st.spinner("Loading the Summary Plot"):
            shap_analyzer.Anal_summary_plot(classes, prediction)

        st.write("Heatmap Plot: allows visualization of the interactions between characteristics and their overall contributions.")
        with st.spinner("Loading the Heatmap"):
            shap_analyzer.Anal_heatmap(classes)

    with col2:

        # Si on a une classification binaire, on calcule puis affiche les mesures de qualités
        if classes.__len__() <= 2:
            # On calcul des mesures de qualité
            fidelity = shap_analyzer.compute_fidelity()
            stability = shap_analyzer.compute_stability()
            

            # On affiche les résultats
            with st.spinner("Loading Fidelity Metric"):
                st.metric("Fidelity", f"{fidelity:.3f}",border=True, help="Correlation between Model predictions and SHAP Explanations")
            with st.spinner("Loading Stability Metric"):
                st.metric("Stability", f"{stability:.3f}",border=True, help="Stability of Explanations when put under a small amount of noise")
        else : 
            st.write("No Quality Metrics available if more than two targets")





