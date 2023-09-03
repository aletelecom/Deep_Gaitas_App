import streamlit as st
import pandas as pd
import numpy as np
import os

import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.graph_objs import Figure

import networkx as nx
from networkx.algorithms import community


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
    G = nx.Graph()

    # Itera sobre las filas del DataFrame
    for index, row in df.iterrows():
        cantantes = row['NER'].split(",")
        agrupacion = row['TITULO']
        for cantante in cantantes:
            if cantante != "":
                if G.has_node(cantante.title()):
                    G.nodes[cantante.title()]['grupos'].add(agrupacion)
                else:
                    G.add_node(cantante.title(), grupos={agrupacion}, color=colores_dict[agrupacion])
        G.add_edges_from([(cantantes[0].title(),node.title()) for node in cantantes[1:]])
        
    return G

@st.cache_data
def graph_plot_plotly(_G) -> Figure:
    """
    Crea y devuelve una figura interactiva de un grafo utilizando la biblioteca Plotly.

    Parámetros:
    -----------
    _G : networkx.Graph
        El grafo que se va a representar. Debe ser un objeto de tipo networkx.Graph.

    Retorna:
    --------
    plotly.graph_objs.Figure
        Una figura interactiva que muestra el grafo con los nodos y aristas coloreados según las comunidades detectadas mediante el algoritmo de modularidad greedy.

    """

    # Encontrar las comunidades utilizando el algoritmo de modularidad greedy
    communities = community.greedy_modularity_communities(_G)

    # Asignar un color a cada comunidad
    community_colors = {}
    color_list = ["blue", "red", "green", "yellow", "orange"]
    for i, c in enumerate(communities):
        community_colors[i] = color_list[i % len(color_list)]

    # Extraer la comunidad de cada nodo
    node_color = [community_colors[i] for i, c in enumerate(communities) for node in c]

    # Posición de los nodos en el diseño
    pos = nx.kamada_kawai_layout(_G)

    # Crear objeto de dispersión para cada nodo
    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', textposition='bottom center',
                            hoverinfo='text', marker=dict(color=node_color, size=10, line_width=1))

    for node, coords in pos.items():
        node_trace['x'] += tuple([coords[0]])
        node_trace['y'] += tuple([coords[1]])
        node_trace['text'] += tuple([node])

    # Crear objeto de dispersión para cada arista
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    for edge in _G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Crear la figura
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Grafo de Agrupaciones Gaiteras',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig


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

# Encuentra las comunidades
communities = community.greedy_modularity_communities(G)

# Crea un diccionario para la asignación de colores a las comunidades
community_colors = {}
color_list = ['blue',
              'red',
              'green',
              'yellow',
              'orange',
              'purple',
              'black',
              'cyan',
              'magenta',
              'gray',
              'brown',
              'm',
              'pink',
              'olive',
              'darkorchid',
              'firebrick',
              'greenyellow',
              'lime',
              'bisque',
              'fuchsia'
             ]

# Asigna los colores a las comunidades
for i,c in enumerate(communities):
    community_colors[i] = color_list[i%len(color_list)]

# Extrae la comunidad de cada nodo
community_node_color = [community_colors[i] for i,c in enumerate(communities) for node in c]

###########################################
# Creación de la página
###########################################

st.title('Grafo global la comunidad gaitera :musical_score:')

with st.expander(
    "Instrucciones", expanded=False
):
    st.write("")
    st.markdown("""
        * Para expandir el grafo, utilizar el símbolo ubicado en la esquina derecha superior.
        * Para volver al tamaño original, solo vuelve a presionarlo.
        * Para alejar y acercar utiliza los botones __+__, y __-__.
        * Con el botón lupa te permite seleccionar un recuadro de los datos para hacer acercamientos dirigidos.
""")
    


# Creamos la figura del grafo
fig = graph_plot_plotly(G)

# Muestra el grafo en la página
st.plotly_chart(
    fig,
    use_container_width=True
    )