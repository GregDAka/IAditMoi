import streamlit as st
from Data import Data

st.sidebar.empty()

st.title("Welcome to IAditMoi")
st.write('''
Welcome, to begin import your data in csv form
''')
st.write('''
-The first line is the attributes line, the last column is the target
''')

# Ici on gère l'import du CSV
upload = st.file_uploader("Choose your data", type="csv")


if upload is not None:
    data = Data(upload)
    st.session_state.data=data
    st.switch_page("pages/main.py")



