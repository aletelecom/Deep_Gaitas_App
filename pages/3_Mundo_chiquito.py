import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

@st.cache_data
# def plot_node_and_connections(_G, node):

#     nivel = 4

#     # Obtiene un subgrafo de los primeros vecinos de n saltos del nodo
#     subgraph_list = []
#     for n in range(1, nivel):
#         subgraph = nx.ego_graph(_G, node, radius=n, center=True)

#         # Establezca el tamaño y el color del nodo en función de su distancia desde el nodo principal
#         node_size = [3000 if nx.shortest_path_length(_G, node, n) == 0 else 50/(nx.shortest_path_length(_G, node, n)) for n in subgraph.nodes()]
#         node_color = [nx.shortest_path_length(_G, node, nivel) for n in subgraph.nodes()]
#         color_map = plt.get_cmap("jet")
#         node_colors = [color_map(c/nivel) for c in node_color]
#         subgraph_list.append(subgraph)

#     fig, axes = plt.subplots(2,2, figsize=(12,12))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

#     for _, ax, subgraph in zip(range(1,nivel), axes.flatten(), subgraph_list):

#         # Dibuja el subgrafo
#         pos = nx.kamada_kawai_layout(subgraph)
#         nx.draw(
#             subgraph,
#             with_labels=False,
#             node_size=node_size,
#             node_color=node_colors,
#             pos=pos,
#             ax=ax
#         )

#         # Agregar una etiqueta solo al nodo principal
#         labels = {node:node}
#         nx.draw_networkx_labels(
#             subgraph,
#             pos,
#             labels=labels,
#             font_size=10,
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=1),
#             ax=ax
#         )

#         # Agregue un cuadro de texto para mostrar el número de nodos en el gráfico
#         text = "Cantidad de nodos: {}".format(len(pos))
#         bbox_props = dict(boxstyle="square,pad=0.3", fc="white", alpha=0.8)
#         ax.text(0.95, 0.05, text, ha="right", va="bottom", transform=ax.transAxes, fontsize=10, bbox=bbox_props)

#         # Agregue una leyenda para codificar el nivel de conexión
#         handles = []
#         for i in range(nivel+1):
#             handles.append(plt.Rectangle((0,0),1,1,fc=color_map(i/nivel)))
#         ax.legend(handles, [f"Nivel {i}" for i in range(nivel+1)], title="Radio de conexión")

#     return fig
# @st.cache_data
def plot_node_and_connections(_G, node, n, _ax):
    # Obtiene un subgrafo de los primeros vecinos de n saltos del nodo
    subgraph = nx.ego_graph(_G, node, radius=n, center=True)

    # Establezca el tamaño y el color del nodo en función de su distancia desde el nodo principal
    node_size = [3000 if nx.shortest_path_length(_G, node, n) == 0 else 50/(nx.shortest_path_length(_G, node, n)) for n in subgraph.nodes()]
    node_color = [nx.shortest_path_length(_G, node, n) for n in subgraph.nodes()]
    color_map = plt.get_cmap("jet")
    node_colors = [color_map(c/n) for c in node_color]

    # Dibuja el subgrafo
    pos = nx.kamada_kawai_layout(subgraph)
    nx.draw(
        subgraph,
        with_labels=False,
        node_size=node_size,
        node_color=node_colors,
        pos=pos,
        ax=_ax
    )

    # Agregar una etiqueta solo al nodo principal
    labels = {node:node}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=1),
        ax=_ax
    )

    # Agregue un cuadro de texto para mostrar el número de nodos en el gráfico
    text = "Cantidad de nodos: {}".format(len(pos))
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", alpha=0.8)
    _ax.text(0.95, 0.05, text, ha="right", va="bottom", transform=_ax.transAxes, fontsize=10, bbox=bbox_props)

    # Agregue una leyenda para codificar el nivel de conexión
    handles = []
    for i in range(n+1):
        handles.append(plt.Rectangle((0,0),1,1,fc=color_map(i/n)))
    _ax.legend(handles, [f"Nivel {i}" for i in range(n+1)], title="Radio de conexión")



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

st.title('Efecto de "Mundo Chiquito" :musical_score:')

st.markdown(
    """
    # Nuestra red de "gaiteros" demuestra ser altamente conectada, tal y como vimos en la sección de __"Centralidad de Cercanía"__. Esta alta conectividad indica una fuerte colaboración y una gran capacidad de interacción entre los "gaiteros" de la red.

Este fenómeno recibe el nombre de __"Efecto de Mundo Pequeño"__ en la cultura pop, el cual ha sido visitado y revisitado varias veces, siendo una de sus instancias más conocidas la frase: "seis grados de separación", la cual generó la publicación de libros, peliculas y demás apariciones en distintos medios de comunicación. El origen del estudio de este fenómeno se atribuye al experimento realizado por __Stanley Milgram__ en la decada de los 60, del cual puedes ahondar en su página de Wikipedia:

[Stanley Milgram](https://es.wikipedia.org/wiki/Stanley_Milgram)
    """
)

with st.expander(
    "Instrucciones", expanded=False
):
    st.write("")
    st.markdown("""
        * Elije a un gaitero para ver su gráfico de mundo pequeño.
""")
    
# Create the figure and axes outside the function
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
nodos = [nodo for nodo in G.nodes()]
sorted(nodos)
node = st.selectbox('Selecciona el tipo de centralidad', nodos)


    # Call the function for each level and subplot
for level, ax in zip(range(1, 5), axes.flatten()):
    plot_node_and_connections(G, node, level, ax)

st.pyplot(fig)