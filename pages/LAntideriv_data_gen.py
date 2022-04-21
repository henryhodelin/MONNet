import os
import numpy as np
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go

import scipy




def linear_antiderivative_training_data_generator(st, **state):

    if "cal_procesos_gaussianos" not in st.session_state:
        st.session_state["cal_procesos_gaussianos"] = False

    st.header("Linear Antiderivative Data Training Generation")

    st.subheader("- Input function generation as gaussian processes: u(x)")
    
    
    with st.expander("See theoretical explanation"):
        
        st.write("""
        En el artículo  DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators exploraron solamente como funciones de entrada u(x) a procesos aleatorios gausianos y polinomios ortogonales de Chebyshev. 
        
        
         - Un proceso gaussiano es un proceso estocástico (una colección de variables aleatórias indexadas por tiempo o espacio), tal que para cada colección finita de dichas variables aleatórias tiene una distribución multivariable normal y que cada combinación de ellas es normalmente distribuida.
       
        Un ejemplo que ayuda a visualizar el caso  del movimiento browniano (movimiento aleatorio de partículas suspendidas en un líquido), y  puede analizarse como una caminada aleatoria (sucesión de pasos aleatorios  en un espacio matemático) . Este  proceso sobre el tiempo puede simularse en una dimensión $d$ comenzando en la posición $0$ moviendo la partícula una distancia aleatoria  $\Delta d$ respecto a su posición previa durante cierto intervalo de tiempo $\Delta t$. Donde la distancia   $\Delta d$ es muestreada de una distribución normal con media $0$ y variancia $\Delta t$. El muestreo de $\Delta d$ de la distribución normal se expresa denota como 
        """)
        st.latex(r"""\Delta d \sim \mathcal{N}""")
        st.write("""
        La figura siguiente muestra una simulación de $5$ trayectorias de un movimiento browniano (también conocidas como realizaciones).""")

        #list_images = os.listdir(os.getcwd()+'/pages/images')
        #st.write(list_images)
        
        MBrowniano = Image.open(os.getcwd()+'/pages/images/'+'Movimiento_Browniano.png')
        st.image(MBrowniano)
        
        st.write(""" Note que en la figura que cada realización corresponde a una función con la forma $d=f(t)$. Esto significa que un proceso estocástico puede interpretarse como una distribución aleatoria sobre funciones (es como escoger aleatoriamente $f(t)$).  """)
        
        st.write("""En específico los procesos gausianos son distribuciones sobre funciones $f(x)$ donde la distribución esta definida por la función media $m(x)$ y una función positiva de covarianza $k(x,x')$, con los valores $x$ de la función y
todos los pares $(x,x')$ posibles en el dominio de entrada: 
        """)
        
        st.latex(r""" f(x) \sim \mathcal{G}(m(x),k(x,x')) """)
        
        st.write(""" donde por cada subconjunto finito $X = \{x_1, \cdots x_n\}$
        del dominio $x$ la distribución marginal (distribución de probabilidad de las variables contenidas en el subconjunto) es una distribución normal multivariante """)
        
        st.write(""" Para muestrear funcioens del proceso gaussiano es necesario definir la media y las funciones de covarianza. Es útil recordar que  la covarianza es una medida de cuánto cambian dos variables juntas, y la función de covarianza, o núcleo, describe la covarianza espacial o temporal de un proceso o campo de variable aleatoria. Es decir la función de covarianza $k(x_a,x_b)$ modela la variabilidad conjunta de las variables aleatorias del proceso gaussiano. Devuelve la covarianza modelada entre cada par $xa$ en
y $x_b$. La función de covarianza tiene que ser una función positivamente definida  """)

        st.write("""
                
        En nuestro caso   las  funciones de entrada  u(x) como campos aleatorios gaussianos con media cero siguiendo las especificaciones del artículo mensionado: 
""")
        st.latex(r""" u  \sim \mathcal{G }\left(m = 0, k_l\left(x_1, x_2\right)\right)""")
        st.latex(r"""k_l(x_1,x_2)= exp(-||x_1-x_2||^2/2l^2) \quad \quad \text{ es el kernel de covarianza} """)
        st.latex(r""" l> 0 \quad \quad \text{ es el  parámetro de escala}""")
        st.write("""
               
        
        
        Para esclarecer los diferentes conceptos relacionados a  los procesos gaussianos es recomendable ir a la página 'https://peterroelants.github.io/posts/gaussian-process-tutorial/#References'
        """)       

    ##################################################################
    
    st.subheader("Definición de la función de correlación")
          
    def exponentiated_quadratic(xa, xb,sigma):
        #"""Exponentiated quadratic  with σ=sigma"""
        # L2 distance (Squared Euclidian)
        sq_norm = -(1/(2*sigma**2)) * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        return np.exp(sq_norm)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        nb_of_samples =st.number_input(label=" Número de  muestras \n de variables independientes X", value=1024)
    with col2:
        values = st.slider('Intervalo de muestreo de X', -3.0, 3.0, (-0.0, 1.0))
        X_min = values[0]
        X_max = values[1]
    with col3:
        sigma = st.slider(label = "Parámetro de escala   " , min_value=0.1, max_value=1.0, value=0.2)
        X = np.expand_dims(np.linspace(X_min, X_max, nb_of_samples), 1)
        COV = exponentiated_quadratic(X, X, sigma)
        
        ver_cov = st.checkbox(label="Visualización de la matriz de covarianza", value=False, key="Ver_COV")
        
    if ver_cov:
        fig, ax1  = plt.subplots( figsize=(3, 3))
        xlim = (X_min, X_max)
        im = ax1.imshow(COV, cmap=cm.YlGnBu)
        cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
        cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
        ax1.set_title((
                'Matriz de covarianza '))
        ax1.set_xlabel('x', fontsize=13)
        ax1.set_ylabel('x', fontsize=13)
        ticks = list(range(int(xlim[0])-1 ,int(xlim[1])+1 ))
        ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
        ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
        ax1.set_xticklabels(ticks)
        ax1.set_yticklabels(ticks)
        ax1.grid(False)
        
        st.pyplot(fig)
            
        st.subheader("Generación de procesos gaussianos")
        
        
    col1, col2 = st.columns(2)
    with col1:
        number_of_functions = st.number_input(label=" Número de  procesos \n a generar ",min_value=1, max_value=10,value=5)
    with col2:
        media = st.number_input(label=" Media  ",min_value=-3.0, max_value=3.0,value=0.0)
            
            
    ############################################################################################
    #                  Función graficar procesos gausianos
    ############################################################################################
    
    col1, col2, col3 = st.columns(3)
    with col1:
        #cal_procesos_gaussianos = st.button(label="GENERAR PROCESOS GAUSSIANOS",  key="Cal_PG")
        st.session_state.cal_procesos_gaussianos = st.button(label="GENERAR PROCESOS GAUSSIANOS",  key="Cal_PG")
    with col2:
        ocult_procesos_gaussianos = st.button(label="OCULTAR PROCESOS GAUSSIANOS",  key="Cal_PG")
    with col3:
        vis_procesos_gaussianos = st.button(label="Visualizar PROCESOS GAUSSIANOS",  key="Cal_PG")

    
    #@st.experimental_memo(max_entries=2)
    def gen_gaussian_processes(mean,COV,number_of_functions):
        # Fetch data from URL here, and then clean it up.
        return np.random.multivariate_normal(mean,cov=COV,size=number_of_functions)
        
    if st.session_state.cal_procesos_gaussianos:
        mean = np.ones(nb_of_samples)*media
        st.session_state.ys = gen_gaussian_processes(mean,COV,number_of_functions)
        #st.write(type(st.session_state.ys.tolist()))
        fig2, ax2  = plt.subplots(figsize=(6, 4))
        for i in range(number_of_functions):
            ax2.plot(X, st.session_state.ys[i], linestyle='-', marker='o', markersize=3)
        ax2.set_xlabel('$x$', fontsize=13)
        ax2.set_ylabel('$u(x)$', fontsize=13)
        ax2.set_title('Procesos gaussianos generados')
        ax2.grid(True)
        st.pyplot(fig2)


    if  ocult_procesos_gaussianos:
        st.session_state.cal_procesos_gaussianos = False

    if  vis_procesos_gaussianos:
        try:
            fig2, ax2  = plt.subplots(figsize=(6, 4))
            for i in range(number_of_functions):
                ax2.plot(X, st.session_state.ys[i], linestyle='-', marker='o', markersize=3)
            ax2.set_xlabel('$x$', fontsize=13)
            ax2.set_ylabel('$u(x)$', fontsize=13)
            ax2.set_title('Procesos gaussianos generados')
            ax2.grid(True)
            st.pyplot(fig2)
        except:
            st.warning('You need to generate the gaussian processes')
        

    ##############################################################################################
    #  Nueva función de generación de procesos gaussianos    
    ##############################################################################################
    #st.write(np.shape(np.squeeze(X))) 
    #st.write(np.shape(st.session_state.ys[0]))

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=np.squeeze(X), y=st.session_state.ys[0],
    #                mode='lines',
    #                name='lines'))
    #st.plotly_chart(fig, use_container_width=True)
    
    
    ##################################################################



    if st.checkbox("Vizualize (Download) code"):
        st.write(""" CODIGO DE GENERACIÓN DE FUNCIÓN DE u(x) COMO PROCESOS GAUSSIANOS""")
        
        code_a = '''
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
         
        ################################################################
        ##     DEFINICIÓN DEL NÚCLEO DE LA FUNCIÓN DE CORRELACION     ##
        ################################################################
        
        def exponentiated_quadratic(xa, xb,sigma):
          """Exponentiated quadratic  with σ=sigma"""
          # L2 distance (Squared Euclidian)
          sq_norm = -(1/(2*sigma)) * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
          return np.exp(sq_norm)
          
        ###################################################
        ##    DEFINICIÓN DE LA FUNCION DE CORRELACIÓN    ##
        ###################################################
        
        nb_of_samples = 512  # Número de puntos donde se va a evaluar
                            # el proceso estocástico
        
        # Muestras de variables independientes 
        X = np.expand_dims(np.linspace(0, 1, nb_of_samples), 1)
        
        # La función  numpy.expand_dims(a, axis)
        # Expand the shape of an array.Insert a new axis that will 
        # appear at the axis position in the expanded array shape.
        # Ej.
        # x = np.array([1, 2])
        # x --> array([[1, 2]])
        # x.shape --> (2,)
        # y = np.expand_dims(x, axis=1)
        # y --> array([[1],
        #              [2]])
        # y.shape --> (2, 1)
        
        # sigma = l -->  
        sigma = 0.2
        COV = exponentiated_quadratic(X, X, sigma)  # Función de correlación
       
             
        ################################################
        ##    GENERACIÓN DE LOS PROCESOS GAUSSIANOS   ##
        ################################################
        
       number_of_functions = 5  # Número de funciones a muestrear (número de procesos a generar)
       mean = np.zeros(nb_of_samples) # Definición de la media (igual a cero)
       ys = np.random.multivariate_normal(mean,cov=COV,size=number_of_functions) # generación de los procesos gaussianos
       
       ###################################################
       ##     VISUALIZACIÓN DE LOS PROCESSOS GENERADOS  ##
       ###################################################
       
       plt.figure(figsize=(6, 4))
       for i in range(number_of_functions):
           plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
       plt.xlabel('$x$', fontsize=13)
       plt.ylabel('$y = f(x)$', fontsize=13)
       plt.title(('5 realizaciones del proceso gaussiano '))
       plt.xlim([-4, 4])
       plt.show()
       
        '''
        st.code(code_a, language='python')

    ###############################################################
    #          - SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL
    ##############################################################




    ##############################################################
    #           - CREACIÓN DE UN ARREGLO DE DATOSS
    ##############################################################




    ##############################################################
    #        - CREACION DE MÚLTIPLES ARREGLOS DE DATOS
    ##############################################################

    





    