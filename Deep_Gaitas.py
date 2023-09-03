import streamlit as st
# from SessionState import _get_state


st.set_page_config(
    page_title='Deep Gaitas',
    page_icon=':musical_score:',
)

from new_pages.Grafo_principal import show_main_graph_page
from new_pages.Estudios_de_comunidad import show_estudio_comunidad
from new_pages.Mundo_chiquito import show_mundo_chiquito
from new_pages.Datos_crudos import show_datos_crudos

st.write("# Bienvenidos a Deep Gaitas! :musical_score:")

st.write('Algunos gráficos pueden tardar un poco en mostrarse.')

pages = [
    'Home',
    'Grafo Principal',
    'Estudio de comunidades',
    'Mundo chiquito',
    'Datos crudos',
    ]
page_to_show = st.selectbox('Seleccione la página a mostrar', pages)

def home_page():
    st.markdown(
        """
        Deep Gaitas es un esfuerzo que busca fomentar el interés del público general en crear una base de datos
        detallada del mundo gaitero, comenzando por las agrupaciones gaiteras y sus miembros.

        Y para que el esfuerzo sea medido, y que sus frutos sean constantes, la mejor manera de continuar es escalonando
        dicho esfuerzo en fases, por lo que __la primera__ fase está compuesta de las siguientes actividades:

        ####    1 Creación de la base de datos, y
        ####    2 Elaboración de los estudios con enfoques en teoría de redes (sociales, colaborativas, de flujos de información).

        Estos estudios nos permitirán conocer aún más este maravilloso mundo de la gaita, su historia, a sus gaiteros y sus interacciones.
        Entre los estudios que podemos realizar con estos datos están:

        * Estudios de colaboraciones entre sus miembros.
        * Estudios de flujos de información dentro de la red de colaboraciones.
        * Conocer qué tan unidos están los miembros de la red.
        * Estudios con ciencia de grafos.
        * Conocer el flujo de talentos entre los distintos grupos gaiteros.

        
        Y muchas más cosas.
    """
)

if page_to_show == 'Home':
    home_page()

elif page_to_show == 'Grafo Principal':
    show_main_graph_page()

elif page_to_show == 'Estudio de comunidades':
    show_estudio_comunidad()

elif page_to_show == 'Mundo chiquito':
    show_mundo_chiquito()

elif page_to_show == 'Datos crudos':
    show_datos_crudos()

else:
    home_page()