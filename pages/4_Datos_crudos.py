import streamlit as st
import pandas as pd
import os


###########################################
# Definición de funciones varias
###########################################

@st.cache_resource
def load_data(path_to_data):
    # Función que permite la carga de los datos.
    # Tiene un decorador que permite dejar los datos en cache
    df = pd.read_excel(path_to_data, index_col=0)
    df.fillna('', inplace=True)

    return df

###########################################
# Instanciación de objetos
###########################################

# Carga los datos
NOMBRE_AGRUPACIONES = 'Agrupaciones_NER_flair(v3).xlsx'
CARPETA_DATASETS = 'Datasets'
path = os.path.join(CARPETA_DATASETS, NOMBRE_AGRUPACIONES)
agrupaciones_df = load_data(path)

###########################################
# Creación de la página
###########################################

st.title('Datos "Ffacrudos" :musical_score:')

st.markdown(
    """
    Para recordar, los datos los obtuvimos de la página de web gaiteros llamada [Sabor Gaitero](http://saborgaitero.com/), la cuál tiene una sección de agrupaciones gaiteras dónde se comenta varias cosas sobre cada agrupación, e indican algunos miembros del mismo.

    Los datos fueron guardados en un archivo MS Excel, con las siguientes caracteristicas:

        * Cada fila es una agrupación gaitera.
        * Dos columnas, una llamada TITULO, otra llamada TEXTO, y otra llamada NER.
        * Esta última contiene los nombres de cada uno de los integrantes de cada grupo gaitero.
    """
)

with st.expander(
    "Instrucciones", expanded=False
):
    st.write("")
    st.markdown("""
        * Elije, o escribe el nombre de un(a) gaitero(a), o músico para ver su gráfico de mundo pequeño.
""")


st.dataframe(agrupaciones_df)