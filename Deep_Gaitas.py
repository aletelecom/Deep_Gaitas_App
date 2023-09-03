import streamlit as st
# from SessionState import _get_state
# from 1_Grafo_principal import show_main_graph_page

st.set_page_config(
    page_title='Deep Gaitas',
    page_icon=':musical_score:',
)

st.write("# Bienvenidos a Deep Gaitas! :musical_score:")

st.markdown(
    """
    Deep Gaitas es un esfuerzo que busca fomentar el interés del público general en crear una base de datos
    detallada del mundo gaitero, comenzando por las agrupaciones gaiteras y sus miembros.

    Y para que el esfuerzo sea medido, y que sus frutos sean constantes, la mejor manera de continuar es escalonando
    dicho esfuerzo en fases, por lo que __la primera__ fase está compuesta de las siguientes actividades:

    ####    - Creación de la base de datos.
    ####    - Elaboración de los estudios con enfoques en redes (sociales, colaborativas, de flujos de información).

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