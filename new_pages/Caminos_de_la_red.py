import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO

import networkx as nx

import matplotlib.cm as cm
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

@st.cache_data
def graph_from_agrupaciones(df:pd.DataFrame, colores_dict:dict) -> nx.Graph:
    """
    Crea y devuelve un grafo a partir de un DataFrame que contiene información sobre agrupaciones musicales y sus miembros.

    Parametros:
    -----------
    df : pandas.DataFrame
        Un DataFrame que contiene información sobre las agrupaciones musicales y sus miembros. Debe tener las columnas 'NER' (nombres de los miembros) y 'TITULO' (nombre de la agrupación).

    colores_dict : dict
        Un diccionario que mapea nombres de agrupaciones a colores correspondientes para su representación en el grafo.

    Retorna:
    --------
    networkx.Graph
        Un grafo creado a partir de los datos proporcionados en el DataFrame. Cada nodo representa un miembro de una agrupación, y los nodos se conectan si pertenecen a la misma agrupación. Los nodos también están coloreados según la agrupación a la que pertenecen.

    """
    # Crea un grafo vacío
    _G = nx.Graph()

    # Itera sobre las filas del DataFrame
    for index, row in df.iterrows():
        cantantes = row['NER'].split(",")
        agrupacion = row['TITULO']
        for cantante in cantantes:
            if cantante != "":
                if _G.has_node(cantante.title()):
                    _G.nodes[cantante.title()]['grupos'].add(agrupacion)
                else:
                    _G.add_node(cantante.title(), grupos={agrupacion}, color=colores_dict[agrupacion])
        _G.add_edges_from([(cantantes[0].title(),node.title()) for node in cantantes[1:]])
        
    return _G

def plot_path(_G, source, target):
    path = nx.shortest_path(_G, source, target)
    pos = nx.kamada_kawai_layout(G)
    
    node_colors = ['lightgrey' if node not in path else 'blue' for node in _G.nodes()]
    node_sizes = [500 if node in (source, target) else 100 if node in path else 100 for node in _G.nodes()]
    
    filtered_edges = [(u, v) for u, v in _G.edges() if u != v]
    edge_colors = ['lightgrey' if not (edge[0] in path and edge[1] in path) else 'red' for edge in filtered_edges]
    edge_width = [0.5 if not (edge[0] in path and edge[1] in path) else 7 for edge in filtered_edges]
    
    # Create a Matplotlib figure
    _, ax = plt.subplots(figsize=(10, 8))

    nx.draw(
        _G,
        pos,
        nodelist=G.nodes(),
        edgelist=filtered_edges,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
        width=edge_width,
        with_labels=False,
        alpha=0.5,
        ax=ax
        )
    
    labels = {node: node for node in path}
    nx.draw_networkx_labels(
        _G,
        pos,
        labels,
        font_color='white',
        bbox={'facecolor':'black', 'edgecolor':'none', 'alpha':0.7},
        font_size=8,
        ax=ax
        )
    
    ax.set_title('Camino entre {}, y {} dentro de la red'.format(source,target), fontsize=15)
    
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png")
    img_bytes.seek(0)
    
    return img_bytes

@st.cache_data
def plot_path_cached(_G, source, target):
    return plot_path(_G, source, target)


###########################################
# Instanciación de objetos
###########################################

# Carga los datos
NOMBRE_AGRUPACIONES = 'Agrupaciones_NER_flair(v3).xlsx'
CARPETA_DATASETS = 'Datasets'
path = os.path.join(CARPETA_DATASETS, NOMBRE_AGRUPACIONES)
agrupaciones_df = load_data(path)

# Crea algunas variables para la ejecución de las funciones
lista_agrupaciones = agrupaciones_df['TITULO']
colores = cm.jet(np.linspace(0, 1, len(lista_agrupaciones)))
colores_dict = dict(zip(lista_agrupaciones,colores))

# Crea el objeto nx.Graph
G = graph_from_agrupaciones(agrupaciones_df, colores_dict)


###########################################
# Creación de la página
###########################################

def show_caminos():

    st.title('Camino entre dos miembros :railway_track:')

    st.markdown(
        """
        Tomando como base el grafo completo de la red, podemos generar un gráfico que nos permita ver el camino que existe entre
        dos miembros de la red, y todos los nodos que ayudan a crear dicho camino.
        """
    )


    with st.expander(
        "Instrucciones", expanded=False
    ):
        st.write("")
        st.markdown("""
            * Elije a dos gaiteros distintos para ver como se interconectan dentro de la red.
    """)
    
    # Creamos un listado de todos los nodos, luego creamos los indices para dejar a dos gaiteras como primera opción en cada
    # objeto "select box".
    gaiteros = [nodo for nodo in G.nodes()]
    default_index_1 = gaiteros.index('Carmencita Silva')
    default_index_2 = gaiteros.index('Carmen Aizpúrua')
    gaitero_1 = st.selectbox('Selecciona el primer gaitero(a)', gaiteros, key=1, index=default_index_1)
    gaitero_2 = st.selectbox('Selecciona el segundo gaitero(a)', gaiteros, key=2, index=default_index_2)

    # Publica la imágen en la app
    img = plot_path_cached(G, gaitero_1, gaitero_2)
    st.image(img)