import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st



st.write('''
# Bienvenue dans un exemple d'application de pédiction avec un model de machine learning
Il s'agit d'un exemple de l'interface elle sera ensuite agrémentée de plusieurs éléments d'éxplicabilités
''')

st.sidebar.header("Les parmaètres d'entrée")

def param_entree():
    longeur_sepal= st.sidebar.slider('Longeur du zizi', 4.3, 7.9, 5.3)
    largeur_sepal= st.sidebar.slider('Largeur du sepal', 2.0, 4.4, 3.3)
    longeur_petal= st.sidebar.slider('Longeur du petal', 1.0, 6.9, 2.3)
    largeur_petal= st.sidebar.slider('Largeur du petal', 0.1, 2.5, 1.3)
    donnees={
        'longeur_sepal':longeur_sepal,
        'largeur_sepal' :largeur_sepal,
        'longeur_petal':longeur_petal,
        'largeur_petal':largeur_petal
    }
    fleur_parametres=pd.DataFrame(donnees,index=[0])
    return fleur_parametres

df_entree=param_entree()

st.subheader('on veut prédire la catégorie de cette fleur')
st.write(df_entree)

def param_sortie():
    iris=datasets.load_iris()
    clf=RandomForestClassifier()
    clf.fit(iris.data,iris.target)

    prediction=clf.predict(df_entree)
    donnees={
        'prediction':prediction,
        'nom_prediction':iris.target_names[prediction]
    }
    resultat_prediction=pd.DataFrame(donnees,index=[0])
    return resultat_prediction

df_sortie=param_sortie()    
st.subheader('Catégorie de la fleut prédit par le model')
tableau_explicatif=pd.DataFrame({'setosa':0,'versicolor':1,'virginica':2},index=[0])
st.write('''Résultats possibles:''',tableau_explicatif,'''Prédiction:''',df_sortie)