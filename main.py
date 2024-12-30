import pandas as pd
from sklearn import datasets
import streamlit as st
from Model import Model
from LinearSVCModel import LinearSVCModel
from SGDclassifierModel import SGDclassifierModel
from Data import Data

# On charge le jeu de donnée (à voir pour le rendre modulaire + tard)
csv_file = "diabete.csv"  # Modifier le nom entre guillemets en fonction du jeu de donnée qu'on veut
data = Data(csv_file)

attributes = data.collectDataAttributes() # On extrait les noms différents attributs du JDD (ex : *taille* de x, *longueur* de x etc)
classes = data.collectDataClasses() # On extrait les différentes classes du JDD (ex : chat, chien, vache)

st.title("Machine Learning Model Explanation Project")
st.write('''
Welcome to an example of a prediction application with a Machine Learning model.
What you're seeing is an example of the interface which will later be further enhanced with several elements of explainability.
For now, only the modularity of the data set is in place although its selection itself isn't yet.
''')

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

# On met en place le modèle de prédiction et on l'effectue
chosenModel = SGDclassifierModel()
model = Model(chosenModel, data, user_inputs_df)
prediction = model.predict()


st.write("The predicted class is :",prediction) # On affiche la prédiction