from concurrent.futures import process
import os
import numpy as np
from PIL import Image

import random
from torch.utils import data


import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go

import scipy
from scipy import spatial
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp

import streamlit as st




def solve_linear_ode(X,U,interval,s0,pts_to_eval):
    X = np.squeeze(X)
    u = interp1d(X, U, kind='cubic')
    model= lambda x, y :  u(x)
    return solve_ivp(model, interval, [s0], method='RK45',t_eval=pts_to_eval,rtol = 1e-5).y[0]
    
@st.experimental_memo
def solve_linear_ode_list(X,interval,s0,pts_to_eval,gaussian_processes_list):
    solutions = [solve_linear_ode(X,U,interval,s0,pts_to_eval) for U in gaussian_processes_list]
    return solutions


        
class GP_gen:
    all_gp_params = []
    # Generacion de un processo gaussiano
    def __init__(self, nb_of_samples, X_min,X_max,sigma, media):
        self.X_min = X_min
        self.X_max = X_max

        # Parametros del proceso gaussiano
        self.sigma = sigma #Parámetro de escala
        self.media = media 
        self.nb_of_samples = nb_of_samples # Número de puntos donde se va a evaluar
                                           # el proceso estocástico
        
        # Creacion de parametros intermedios
        self.X, self.mean = self.gp_params_processor()
        self.COV = self.exponentiated_quadratic()
        
        # Generacion del proceso gaussiano
        #self.gp = np.random.multivariate_normal(self.mean,cov=self.COV,size=1)
        self.gp = self.gp_generator()
        
        #Guardar todos los  
        self.all_gp_params.append(self)

    def exponentiated_quadratic(self):
        #"""Exponentiated quadratic  with σ=sigma"""
        # # L2 distance (Squared Euclidian)
        sq_norm = -(1/(2*self.sigma**2)) * scipy.spatial.distance.cdist(self.X, self.X, 'sqeuclidean')
        return np.exp(sq_norm)

    def gp_params_processor(self):
        X = np.expand_dims(np.linspace(self.X_min, self.X_max, self.nb_of_samples), 1)
        mean = np.ones(self.nb_of_samples)*self.media
        return X, mean

    def gp_generator(self):
        return np.random.multivariate_normal(self.mean,cov=self.COV,size=1)



        



class Data_gen(GP_gen):
    all_data = []
    # Generacion de un arreglo de datos
    def __init__(self, nb_of_samples, X_min,X_max,sigma, media,interval,s0,pts_to_eval):
        GP_gen.__init__(self, nb_of_samples, X_min,X_max,sigma, media)
        self.interval = interval
        self.s0 = s0
        self.pts_to_eval = pts_to_eval
        self.s = solve_linear_ode(self.X,self.gp,self.interval,s0,pts_to_eval)

        #Data_gen.all_data.append(self)
        self.all_data.append(self)

    # def solve_linear_ode(X,U,interval,s0,pts_to_eval):
    #     X = np.squeeze(X)
    #     u = interp1d(X, U, kind='cubic')
    #     model= lambda x, y :  u(x)
    #     return solve_ivp(model, interval, [s0], method='RK45',t_eval=pts_to_eval,rtol = 1e-5).y[0]
    






def linear_antiderivative_OPP_training_data_generator(st, **state):

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
    
        
        
        MBrowniano = Image.open(os.getcwd()+'/pages/images/'+'Movimiento_Browniano.png')
        st.image(MBrowniano)
        
        st.write(""" Note que en la figura que cada realización corresponde a una función con la forma $d=f(t)$. Esto significa que un proceso estocástico puede interpretarse como una distribución aleatoria sobre funciones (es como escoger aleatoriamente $f(t)$).  """)
        
        st.write("""En específico los procesos gausianos son distribuciones sobre funciones $f(x)$ donde la distribución esta definida por la función media $m(x)$ y una función positiva de covarianza $k(x,x')$, con los valores $x$ de la función y
todos los pares $(x,x')$ posibles en el dominio de entrada: 
        """)
        
        st.latex(r""" f(x) \sim \mathcal{G}(m(x),k(x,x')) """)
        
        st.write(""" donde por cada subconjunto finito $X = \{x_1, \cdots x_n\}$
        del dominio $x$ la distribución marginal (distribución de probabilidad de las variables contenidas en el subconjunto) es una distribución normal multivariante """)
        
        st.write(""" Para muestrear funcioens del proceso gaussiano es necesario definir la media y las funciones de covarianza. Es útil recordar que  la covarianza es una medida de cuánto cambian dos variables juntas, y la función de covarianza, o núcleo, describe la covarianza espacial o temporal de un proceso o campo de variable aleatoria. Es decir la función de covarianza $k(x_a,x_b)$ modela la variabilidad conjunta de las variables aleatorias del proceso gaussiano. Devuelve la covarianza modelada entre cada par $x_a$ en
y $x_b$. La función de covarianza tiene que ser una función positivamente definida  """)
    
        st.write("""
                
        En nuestro caso   las  funciones de entrada  u(x) como campos aleatorios gaussianos con media cero siguiendo las especificaciones del artículo mensionado: 
#""")
        st.latex(r""" u  \sim \mathcal{G }\left(m = 0, k_l\left(x_1, x_2\right)\right)""")
        st.latex(r"""k_l(x_1,x_2)= exp(-||x_1-x_2||^2/2l^2) \quad \quad \text{ es el kernel de covarianza} """)
        st.latex(r""" l> 0 \quad \quad \text{ es el  parámetro de escala}""")
        st.write("""
               
        
        
        Para esclarecer los diferentes conceptos relacionados a  los procesos gaussianos es recomendable ir a la página 'https://peterroelants.github.io/posts/gaussian-process-tutorial/#References'
        """)       

    ##################################################################
    
    st.subheader("Definición de la función de correlación")
          
        
    col1, col2, col3 = st.columns(3)
    with col1:
        nb_of_samples =st.number_input(label=" Número de  muestras \n de variables independientes X", value=1024)
    with col2:
        values = st.slider('Intervalo de muestreo de X', -3.0, 3.0, (-0.0, 1.0))
        X_min = values[0]
        X_max = values[1]
    with col3:
        sigma = st.slider(label = "Parámetro de escala   " , min_value=0.1, max_value=1.0, value=0.2)

        
        ver_cov = st.checkbox(label="Visualización de la matriz de covarianza", value=False, key="Ver_COV")
    
    
    col1, col2 = st.columns(2)
    with col1:
        number_of_functions = st.number_input(label=" Número de  procesos \n a generar ",min_value=1, max_value=20,value=10)
    with col2:
        media = st.number_input(label=" Media  ",min_value=-3.0, max_value=3.0,value=0.0)

    gaussian_process_params = GP_gen(nb_of_samples,X_min,X_max,sigma,media)
    
    if ver_cov:
        fig, ax1  = plt.subplots( figsize=(3, 3))
        xlim = (gaussian_process_params.X_min, gaussian_process_params.X_max)
        im = ax1.imshow(gaussian_process_params.COV, cmap=cm.YlGnBu)
        cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
        cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
        ax1.set_title((
                'Matriz de covarianza '))
        ax1.set_xlabel('x', fontsize=13)
        ax1.set_ylabel('x', fontsize=13)
        ticks = list(range(int(xlim[0])-1 ,int(xlim[1])+1 ))
        ax1.set_xticks(np.linspace(0, len(gaussian_process_params.X)-1, len(ticks)))
        ax1.set_yticks(np.linspace(0, len(gaussian_process_params.X)-1, len(ticks)))
        ax1.set_xticklabels(ticks)
        ax1.set_yticklabels(ticks)
        ax1.grid(False)
        
        st.pyplot(fig)
            
        st.subheader("Generación de procesos gaussianos")
        
        
    
            
            
    ############################################################################################
    #                  Función graficar procesos gausianos
    ############################################################################################
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        #cal_procesos_gaussianos = st.button(label="GENERAR PROCESOS GAUSSIANOS",  key="Cal_PG")
        st.session_state.cal_procesos_gaussianos = st.button(label="GENERAR PROCESOS GAUSSIANOS",  key="Cal_PG")
    with col2:
        delete_procesos_gaussianos = st.button(label="ELIMINAR PROCESOS GAUSSIANOS",  key="Del_PG") 
    with col3:
        ocult_procesos_gaussianos = st.button(label="OCULTAR PROCESOS GAUSSIANOS",  key="Cal_PG")
    with col4:
        vis_procesos_gaussianos = st.button(label="Visualizar PROCESOS GAUSSIANOS",  key="Cal_PG")

    

    @st.experimental_memo
    def plot_pg(ys):
        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(go.Scatter(x=np.squeeze(gaussian_process_params.X) , y=ys[i][0] ,
                        mode='lines+markers',
                        name='u_'+str(i)))
        fig.update_layout(xaxis_title='x',
                   yaxis_title='u(x)')
        return fig 
    
    
    if st.session_state.cal_procesos_gaussianos:
        multiple_gaussian_processes =  [GP_gen(nb_of_samples,X_min,X_max,sigma,media) for item in range(number_of_functions) ] 
        processes = [intance.gp for intance in GP_gen.all_gp_params]
        st.session_state.ys = processes
        st.info('Gaussian Processes Generated')
        fig = plot_pg(st.session_state.ys)
        st.plotly_chart(fig, use_container_width=True)
        
    if delete_procesos_gaussianos:
        GP_gen.all_gp_params = []
    
    if  vis_procesos_gaussianos:
        try:
            st.info('Gaussian Processes Generated')
            fig = plot_pg(st.session_state.ys)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning('You need to generate the gaussian processes')
        

    

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

    ################################################################
    ##           SELECCIÓN DE LAS FUNCIONES DE ENTRADA 
    ################################################################

    u_options = []
    u_num_options = []
    u_num = np.linspace(0,number_of_functions,number_of_functions+1)
    for j in range(number_of_functions):
        u_options.append("u_"+str(int(j) ) +"(x)")
            
    st.subheader('INPUTS FUNCTIONS u(x) TO SELECT ')        
    u_seleccionada  = st.multiselect(label=" ", options=u_options)
            
            
    try:
        u_index = []
        for k in u_seleccionada:
            u_index.append(u_options.index(k) )
        st.subheader('INPUT FUNCTIONS SELECTED')
        fig = go.Figure()
        for i in u_index:
            fig.add_trace(go.Scatter(x=np.squeeze(gaussian_process_params.X), y=st.session_state.ys[i][0],
                    mode='lines+markers',
                    name=u_options[i]))
        fig.update_layout(xaxis_title='x',
                yaxis_title='u_i(x)')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning('SELECT AL LEAST ONE FUNCTION')
    
    ##############################################################
    #          - SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL
    #            UTILIZANDO List Comprehension
    ##############################################################
   
    st.subheader("ODEs SOLUTIONS CALCULATIONS")
    
    
    
    col1, col2 = st.columns([1,1])
    with col1:
        s0 = st.slider('ODE initial condition s(0)', 0.0, 1.0)

       

    #col1, col2 = st.columns([1,2])
    #with col1:
        
    #with col2:
    #    if st.checkbox("EXAMPLE CODE"):
    #        code_b = """ 

    
    #        """

    #st.write(interval)

    ###########################################################################

    if st.checkbox("CALCULATE"):
        interval = [0,1]
        X_eval = np.linspace(0,1,100)#np.squeeze(gaussian_process_params.X)
        l_test = solve_linear_ode_list(gaussian_process_params.X,interval,s0,X_eval,st.session_state.ys)
        
        fig = go.Figure()
        for i in u_index:
            fig.add_trace(go.Scatter(x=np.squeeze(gaussian_process_params.X), y=l_test[i],
                    mode='lines+markers',
                    name=u_options[i]))
        fig.update_layout(xaxis_title='x',
                yaxis_title='s_i(x)')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training Data Generation")
    with st.expander("View training data format"):
        st.write("En construccion")


    if st.checkbox("Creating training data"):
        interval = [0,1]
        X_eval = np.linspace(0,1,100)

        training_data = Data_gen(nb_of_samples, X_min,X_max,sigma, media,interval,s0,X_eval)
        st.write([training_data.all_gp_params[1].pts_to_eval,training_data.all_gp_params[1].s])
        
         
        




    
