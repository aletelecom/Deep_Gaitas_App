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


def plot_graph_with_centrality(_G, centralidad, layout, alpha_global=0.7):
    """
    Función que genera un gráfico a partir de un objeto "Grafo" de NetworkX.
    
    Params:
        * G -> un objeto NetworkX graph.
        * centralidad -> un objeto cadena de texto indicando el tipo de centralidad a graficar.
        * layout -> un objeto cadena de texto que indica el tipo de algoritmo de posicionamiento
        que tendrán los nodos del grafo.
        * ax -> un objeto Axes de Matplotlib.
        * alpha_global -> un dato tipo "float" que indica la transparencia del dibujo de red.
        
    Returns:
        * Un gráfico del grafo.
    """
    # Calcula la centralidad de grado del grafo
    if centralidad == 'Intermediación':
        centrality = nx.betweenness_centrality(_G)
        titulo = 'Intermediación'
    elif centralidad == 'Grado':
        centrality = nx.degree_centrality(_G)
        titulo = 'Grado'
    elif centralidad == 'Cercanía':
        centrality = nx.closeness_centrality(_G)
        titulo = 'Cercanía'
    elif centralidad == 'Vector Propio':
        centrality = nx.eigenvector_centrality(_G)
        titulo = 'Vector Propio'
    else:
        centrality = nx.betweenness_centrality(_G)
    
    # Ajusta el tamaño de los nodos según su centralidad de grado
    node_sizes = [centrality[node] * 10000 for node in _G.nodes()]
    
    # Ajusta el color de los nodos según su centralidad de grado
    node_colors = [centrality[node] for node in _G.nodes()]
    
    if layout == 'Kawai':
        pos = nx.kamada_kawai_layout(_G)
#         pos['Ricardo Aguirre'] += (0.4, 0.4)
    elif layout == 'Circular':
        pos = nx.circular_layout(_G)
    elif layout == 'Spring':
        pos = nx.spring_layout(_G)
    elif layout == 'Spectral':
        pos = nx.spectral_layout(_G)
    else:
        pos = nx.kamada_kawai_layout(_G)
    
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibuja el gráfico
    nx.draw(
        _G,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.jet,
        with_labels=False,
        pos=pos,
        ax=ax,
        alpha=alpha_global,
    )
    
    # Añade las etiquetas a los nodos del top 10 de ranking
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    top_10 = sorted_nodes[:10]
    labels = {node: node for node in top_10}
    
    # Dibuja las etiquetas
    nx.draw_networkx_labels(
        _G,
        pos,
        labels,
        font_size=8,
        ax=ax,
        alpha=alpha_global
    )
    for node in top_10:
        x, y = pos[node]
        plt.text(
            x,
            y,
            node,
            bbox=dict(
                facecolor='white',
                alpha=alpha_global,
            ),
                 ha="center",
                 va="center"
        )

    ax.set_title('Colaboraciones Gaiteras, resaltando la "Centralidad de {}"'.format(titulo), fontsize=15)
    
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

alpha_global = 0.7

# Crea el objeto nx.Graph
G = graph_from_agrupaciones(agrupaciones_df, colores_dict)
centrality_selection = [
    'Intermediación',
    'Grado',
    'Cercanía',
    'Vector Propio'
    ]
layout_selection = [
    'Kawai',
    'Circular',
    'Spring',
    'Spectral',
]

# Las explicaciones de cada medida de centralidad se guardan en un diccionario que luego será utilizado
# para mostrar cada una según la elección del usuario.
explicaciones_centralidades = {
    'Intermediación':"""
    # Centralidad de intermediación (Betweenness Centrality)

    La centralidad de intermediación nos indica qué tan importante es un nodo en el flujo de información de un grafo, debido a que mide la cantidad de veces que un nodo es incluido en todos los caminos de una red.

    En pocas palabras: un gaitero con una centralidad de intermediación elevada indica que ella/él es importante para el flujo de la información en la red bajo estudio, debido a que la información que se origine en cualquier parte de la red tiene una alta probabilidad de pasar por ella/él.
""",

    'Grado': """
    # Centralidad de grado (Degree Centrality)
    En pocas palabras, mide la cantidad de conexiones de un nodo (gaitero), por lo que entre más alto el numero, más cantidad de conexiones o colaboraciones tiene un gaitero.
""",

    'Cercanía':"""
    # Centralidad de cercanía (Closeness Centrality)

    La centralidad de cercanía es un indicativo de cuán cercano un nodo está del resto de los nodos en un grafo, es decir, que un nodo con alta centralidad de cercanía puede distribuir la información a muchos nodos de manera rápida, y fácil.

    Otra manera de ver esta medida es que los nodos (gaiteros) con una alta medida de __"Centralidad de Cercanía"__ son los nodos con los caminos más cortos hacia ella/él desde el resto de los miembros de la red.
""",

    'Vector Propio' : """
    # Centralidad de vector propio (Eigenvector Centrality)

    La centralidad de vector propio indica que tan bien conectado está un nodo, es decir, que tan importante son los nodos vecinos del nodo bajo estudio, dándonos una idea de la calidad de conexiones de un nodo.
"""
}

###########################################
# Creación de la página
###########################################

st.title('Medidas de centralidad de la red Gaitera :musical_score:')

st.markdown(
    """
    # Medidas de centralidad

    Las medidas de centralidad son una herramienta fundamental en el análisis de grafos y tienen aplicaciones en diferentes ámbitos, desde la investigación social hasta la investigación en tecnología de la información. En esta sección, nos enfocaremos en el uso de las medidas de centralidad en el análisis de grafos de colaboración entre los distintos Gaiteros. A través de la identificación de los nodos centrales en este grafo, podemos tener una comprensión más profunda de la estructura y dinámica de la colaboración en este sector específico. En esta sección, profundizaremos en las medidas de centralidad y cómo se aplican en el contexto de los grafos de colaboración musical.

    Si quieres ahondar en los conceptos matematicos de cada medida de centralidad, te dejo el enlace wikipedia del tema:

    [Medidas de centralidad (Wikipedia)](https://es.wikipedia.org/wiki/Centralidad)
    """
)


with st.expander(
    "Instrucciones", expanded=False
):
    st.write("")
    st.markdown("""
        * Elije una de las opciones de medida de centralidad, y
        * Elije una de las distribuciones de la posición de los nodos.
""")
    
centralidad = st.selectbox('Selecciona el tipo de centralidad', centrality_selection)
layout = st.selectbox('Selecciona la distribución de los nodos', layout_selection)

fig = plot_graph_with_centrality(G, centralidad, layout)
st.pyplot(fig, use_container_width=True)

st.markdown(explicaciones_centralidades[centralidad])