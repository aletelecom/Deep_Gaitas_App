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
from new_pages.Caminos_de_la_red import show_caminos

###########################################
# Creación de la página
###########################################

st.write("# Bienvenidos a Deep Gaitas! :musical_score:")

st.write('Algunos gráficos pueden tardar un poco en mostrarse.')

pages = [
    'Home',
    'Grafo Principal',
    'Estudio de comunidades',
    'Mundo chiquito',
    'Caminos dentro de la red',
    'Datos crudos',
    'Información',
    ]
page_to_show = st.selectbox('Seleccione la página a mostrar', pages)

def home_page():
    st.markdown(
        """
        Deep Gaitas es la combinación de varias herramientas estudio de redes sociales, y de los datos
        de una red de colaboraciones entre músicos, compositores, y escritores del género músical llamado
        __Gaita Zuliana__.

        Te invito a explorar esta increíble combinación de dos mundos, y las maravillas que arroja.
        """
    )
    
def info_page():
    st.markdown(
            """
            Deep Gaitas es un esfuerzo con el que busco fomentar el interés del público general en crear una base de datos
            detallada del mundo gaitero, comenzando por las agrupaciones gaiteras y sus miembros.

            Y para que el esfuerzo sea medido, y que sus frutos sean constantes, la mejor manera de continuar es escalonando
            dicho esfuerzo en fases, por lo que __la primera__ fase está compuesta de las siguientes actividades:

            ####    1 Creación de la base de datos.
            ####    2 Elaboración de los estudios con enfoques en teoría de redes (sociales, colaborativas, de flujos de información).

            Estos estudios nos permitirán conocer aún más este maravilloso mundo de la gaita, su historia, a sus gaiteros y sus interacciones.
            Entre los estudios que podemos realizar con estos datos están:

            * Estudios de colaboraciones entre sus miembros.
            * Estudios de flujos de información dentro de la red de colaboraciones.
            * Conocer qué tan unidos están los miembros de la red.
            * Estudios con ciencia de grafos.
            * Conocer el flujo de talentos entre los distintos grupos gaiteros.

            
            Y muchas más cosas.

            ## Tecnicalidades

            La base de datos con la que creé esta app se encuentra lejos de estar depurada, sin embargo presenta un excelente punto
            de partida para implementar los estudios y algorítmos de redes que son presentados.
            
            Entre los temas que presenta la base de datos les puedo comentar:

            * Falta de depuración __NER__: hay algunos gaiteros o músicos que están con sus nombres, y adicionalmente
            están con sus sobrenombres (motes, alias, etc), lo que produce una duplicación de información, y por ende, una falla en las relaciones
            globales resultantes.
            * Información incompleta: De ninguna manera las representaciones, conlcusiones, y resultados mostrados en la app, y el estudio
            pretenden ser definitivas, los gaiteros, y agrupaciones que conforman los datos del estudio no son todos los que existen, pero eso
            es exactamente lo que persigo creando este estudio, que como comunidad creemos una base de datos definitiva y depurada por todos
            los zulianos.

            ## Contacto

            Si deseas colaborar, en esta tarea, no dudes en contactarme a mi correo electrónico:

            aletelecom.medina@gmail.com
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

elif page_to_show == 'Caminos dentro de la red':
    show_caminos()

elif page_to_show == 'Datos crudos':
    show_datos_crudos()

elif page_to_show == 'Información':
    info_page()

else:
    home_page()