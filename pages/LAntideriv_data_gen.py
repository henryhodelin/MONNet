from logging import exception
import os
from turtle import position
import numpy as np
from PIL import Image

import random


import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go

import scipy
from scipy import spatial
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp





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

    
    def gen_gaussian_processes(mean,COV,number_of_functions):
        return np.random.multivariate_normal(mean,cov=COV,size=number_of_functions)
        
    if st.session_state.cal_procesos_gaussianos:
        mean = np.ones(nb_of_samples)*media
        st.session_state.ys = gen_gaussian_processes(mean,COV,number_of_functions)
        #st.write(type(st.session_state.ys.tolist()))
        st.subheader('Gaussian Processes Generated')
        fig = go.Figure()
        for i in range(number_of_functions):
            fig.add_trace(go.Scatter(x=np.squeeze(X), y=st.session_state.ys[i],
                        mode='lines+markers',
                        name='u_'+str(i)))
        fig.update_layout(xaxis_title='x',
                   yaxis_title='u(x)')
        st.plotly_chart(fig, use_container_width=True)
        

    if  ocult_procesos_gaussianos:
        st.session_state.cal_procesos_gaussianos = False

    if  vis_procesos_gaussianos:
        try:
            st.subheader('Gaussian Processes Generated')
            fig = go.Figure()
            for i in range(number_of_functions):
                fig.add_trace(go.Scatter(x=np.squeeze(X), y=st.session_state.ys[i],
                        mode='lines+markers',
                        name='u_'+str(i)))
            fig.update_layout(xaxis_title='x',
                   yaxis_title='u_i(x)')
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

    ###############################################################
    #           SELECCIÓN DE LAS FUNCIONES DE ENTRADA 
    ###############################################################

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
            fig.add_trace(go.Scatter(x=np.squeeze(X), y=st.session_state.ys[i],
                    mode='lines+markers',
                    name=u_options[i]))
        fig.update_layout(xaxis_title='x',
                yaxis_title='u_i(x)')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning('SELECT AL LEAST ONE FUNCTION')
    
    ###############################################################
    #          - SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL
    #            UTILIZANDO List Comprehension
    ##############################################################
   
    st.subheader("ODEs SOLUTIONS CALCULATIONS")
    
    

    @st.experimental_memo
    def solve_linear_ode(X,U,interval,s0_ode_lineal,X_eval):
        X = np.squeeze(X)
        u = interp1d(X, U, kind='cubic')
        model= lambda x, y :  u(x)
        return solve_ivp(model, interval, [s0_ode_lineal], method='RK45',t_eval=X_eval,rtol = 1e-5).y[0]
    
    col1, col2 = st.columns([1,1])
    with col1:
        s0_ode_lineal = st.slider('ODE initial condition s(0)', 0.0, 1.0)
    

    col1, col2 = st.columns([1,2])
    with col1:
        if st.checkbox("CALCULATE"):
            interval = [0,1]
            X_eval = np.squeeze(X)
            l_test = [solve_linear_ode(X,U,interval,s0_ode_lineal,X_eval) for U in st.session_state.ys]
    with col2:
        if st.checkbox("EXAMPLE CODE"):
            code_b = """
import numpy as np

import plotly.graph_objects as go

import scipy
from scipy.interpolate import interp1d
from scipy.integrate import  solve_ivp

#############################################################
###          IMPLEMENTATION OF THE NUMERICAL   
###       SOLUTION OF THE DIFFERENTIAL EQUATION
###       dy/dx = u(x) = cos(2*pi*x) 
##############################################################

X_min = 0
X_max = 1
nb_of_samples = 100


X = np.expand_dims(np.linspace(X_min, X_max, nb_of_samples), 1)
U = np.cos(2*np.pi*X)

interval = [0,1]               # CALCULATION DOMAIN
s0_ode_lineal = 0              # INITIAL CONDITION
X_eval = np.squeeze(X)         # POINTS OF EVALUATION 



def solve_linear_ode(X,U,interval,s0_ode_lineal,X_eval):
    X = np.squeeze(X)
    u = interp1d(X, U, kind='cubic')
    model= lambda x, y :  u(x)
    return solve_ivp(model, interval, [s0_ode_lineal], method='RK45',t_eval=X_eval,rtol = 1e-5).y[0]

solution = solve_linear_ode(X,U,interval,s0_ode_lineal,X_eval)


fig = go.Figure()
fig.add_trace(go.Scatter(x=np.squeeze(X), y=np.squeeze(U),
        mode='lines',
        name='u(x)'))
fig.add_trace(go.Scatter(x=np.squeeze(X), y=np.squeeze(solution),
        mode='lines+markers',
        name='y(x)'))        
fig.update_layout(xaxis_title='x',
                yaxis_title='Functions')
fig.show()
        """
            st.code(code_b, language='python')




    
        #st.write(len(l_test)) 
    
    st.subheader("VISUALIZATION OF THE SOLUTION FOR THE INPUT FUNCTIONS SELECTED")
    
    try:
        fig = go.Figure()
        for i in u_index:
            fig.add_trace(go.Scatter(x=np.squeeze(X), y=l_test[i],
                    mode='lines+markers',
                    name=u_options[i]))
        fig.update_layout(xaxis_title='x',
                yaxis_title='u_i(x)')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning('SELECT AL LEAST ONE FUNCTION')
    


    ##############################################################
    #           - CREACIÓN DE UN ARREGLO DE DATOS
    ##############################################################

    
    x = np.squeeze(X)
    u = st.session_state.ys

    @st.experimental_memo
    def data_training_array_gen(X,U,interval,s0_ode_lineal,X_eval):
        u = interp1d(X, U, kind='cubic')
        model= lambda x, y :  u(x)
        return U, np.expand_dims(s0_ode_lineal,0) , X_eval, solve_ivp(model, interval, [s0_ode_lineal], method='RK45',t_eval=X_eval,rtol = 1e-5).y[0]
    

    st.subheader("Generation of arrays of training data") 

    col1, col2, col3 = st.columns(3)
    if "counter" not in st.session_state:
        st.session_state["counter"] = 0

    @st.experimental_memo
    def init_gen(number_of_functions,counter):
        positions = [random.random() for j in range(number_of_functions)]
        initial_conditions = [random.random() for j in range(number_of_functions)]
        return positions, initial_conditions

    if "G_data" not in st.session_state:
        st.session_state["G_data"] = False 

    if "training_data" not in st.session_state:
        st.session_state["training_data"] = []


    with col1:
        st.session_state.G_data = st.button(label="GENERATE TRAINING DATA",  key="Cal_GTD")

    with col2:
        hide_data = st.button(label="HIDE DATA")
        
    with col3:
        view_data = st.button(label="VIEW DATA")
        
    if st.session_state.G_data:
        #positions = [random.random() for j in range(number_of_functions)]
        # #initial_conditions = [random.random() for j in range(number_of_functions)]
        positions, initial_conditions = init_gen(number_of_functions,st.session_state.counter)
        try:
            st.session_state.training_data = [data_training_array_gen(x,u,interval,initial_conditions,np.expand_dims(positions,0) ) for  (u,initial_conditions,positions) in zip(u,initial_conditions,positions)]
        except:
             st.info("Need to generate the input functions")

        st.write(st.session_state.training_data)
        st.session_state.counter += 1

    if hide_data:
        st.session_state.G_data = False

    if view_data:
        try:
            st.write(st.session_state.training_data)
        except:
            st.info("Need to generate arrays of training data")

    if st.checkbox("EXAMPLE CODE",key = 'c'):
        code_c = """
        
        """

        st.code(code_c, language='python')




    #st.write("lista de parametros")
    
    
    
    #st.write('posiciones de evaluacion =',positions)
    #st.write('elemento de posiciones de evaluacion type =',type(np.expand_dims(positions[0],0) ))


    #st.write('condiciones iniciales =',initial_conditions)

    #st.write("#################################################")

    
    

    #st.write("Evaluacion exitosa de un punto")

    #st.write("lista de parametros")

    #st.write("x type =",type(x))
    #st.write( "x shape =",np.shape(x))

    #st.write("u type =",type(st.session_state.ys[0]))
    #st.write( "u shape =",np.shape(st.session_state.ys[0]))
    
    #st.write("s0 type = ", type(s0_ode_lineal)) 
    #st.write("s0 shape =", np.shape(s0_ode_lineal))

    #st.write("x_eval type = ",type(x))
    #st.write("x_eval shape =", np.shape(x))

    

    #single_training_data = data_training_array_gen(x,st.session_state.ys[0],interval,s0_ode_lineal,x)
    
    #st.write("Salida:")

    #st.write(type(single_training_data))
    #st.write(type(single_training_data[0]))
    
    #st.write(single_training_data[0][1])
    #st.write(single_training_data[0][2])

    #st.write("#################################################")

  


    #training_data = [data_training_array_gen(x,u,interval,initial_conditions,np.expand_dims(positions,0) ) for  (u,initial_conditions,positions) in zip(u,initial_conditions,positions)]

    #st.write(training_data)



    





    