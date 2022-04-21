import streamlit as st

import numpy as np

import random

import scipy
from scipy import spatial
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp

import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image

if "visualizar_ocultar_generacion_u" not in st.session_state:
    st.session_state["visualizar_ocultar_generacion_u"] = False

if "visualizar_ocultar_generacion_u_code" not in st.session_state:        
    st.session_state["visualizar_ocultar_generacion_u_code"] = False
    
if "test_u_gen_code" not in st.session_state:
    st.session_state["test_u_gen_code"] = False

    
if "ys" not in st.session_state:
    st.session_state["ys"] = 0.0
    
    
if "visualizar_ocultar_solucion_eq_diferencial" not in st.session_state:    
    st.session_state["visualizar_ocultar_solucion_eq_diferencial"] = False
    
if "visualizar_ocultar_ode_lineal_codigo" not in st.session_state:        
    st.session_state["visualizar_ocultar_ode_lineal_codigo"] = False

    
st.header("Generación de Datos MultiOpONet")

with st.expander("Objetivo:"):
    st.write("""
    El primer paso a la hora de entrenar una red neuronal consiste en entrenar una red neuronal y nuestro objetivo es crear el conjunto de funciones necesarias para completar esta tarea
    
    Como queremos generar una red neuronal capaz de representar múltiples operadores para completar este paso sera necesario: 
    
    - Utilizar un método numérico para generar un arreglo de datos de entrenamiento
    - Utilizar un ciclo para generar un número los suficientemente grande de arreglos de entrenamiento
    - Crear una función que permita la selección de lotes de datos para facilitar el proceso de entrenamiento
    
    """)
    
with st.expander("Descripción de operadores múltipes a representar:"):
    st.write("""
    Los operadores a representar seran similares a los operadores del artículo  DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators, que serían:
    
    """)
    st.write("""
    El operador que mapea del espacio de funciones u(x) al espacio de funciones s(x) cuando 
    
    a) ODE LINEAL
    """)
    st.latex(r'''
    \frac{ds(x)}{dx} = u(x) \quad s(0) = [0,1]
    ''')
    
    st.write("""
    b) ODE NO LINEAL
    """)
    st.latex(r"""
    \frac{ds(x)}{dx} = -s^2(x) + u(x) \quad s(0) = [0,1]
    """)
    
    st.write("""
    c) SISTEMA ODE NO LINEAL
    """)
    st.latex(r"""
    \frac{ds_1(x)}{dx} = s_2(x)
    \quad \quad 
    \frac{ds_2(x)}{dx} = -ksin(s_1) + u(t)
    \quad \quad \quad s_1(0) = [0,1] \quad s_2(0) = [0,1]
    """)
    st.write("""
    d) PDE NO LINEAL
    """)
    st.latex(r"""
    \frac{1}{2}\frac{\partial s_(x,z)}{dx^2} + i \frac{\partial a}{\partial z} + u(x)|s(x,z)|^2 s(x,z)= 0    \quad \quad s(x,0) = v(x)e^{iz}
    """)

    
    
with st.expander("Generación de arreglo de datos un arreglo de datos de entrenamiento:"):
    st.write("""
    El proceso de generación de una arreglo de datos de entrenamiento se puede dividir en las siguientes partes:
    
    
    """)
    
    st.write("""
    - GENERACIÓN DE LA FUNCIÓN u(x)
    """)    
    VO_U = st.button("Visualizar/Ocultar Explicación teórica",key="buttom VO_U")
    
        
    if VO_U:
        st.session_state.visualizar_ocultar_generacion_u = not st.session_state.visualizar_ocultar_generacion_u 
        
    if st.session_state.visualizar_ocultar_generacion_u:
        st.write("""
        En el artículo  DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators exploraron solamente como funciones de entrada u(x) a procesos aleatorios gausianos y polinomios ortogonales de Chebyshev. 
        
        
         - Un proceso gaussiano es un proceso estocástico (una colección de variables aleatórias indexadas por tiempo o espacio), tal que para cada colección finita de dichas variables aleatórias tiene una distribución multivariable normal y que cada combinación de ellas es normalmente distribuida.
       
        Un ejemplo que ayuda a visualizar el caso  del movimiento browniano (movimiento aleatorio de partículas suspendidas en un líquido), y  puede analizarse como una caminada aleatoria (sucesión de pasos aleatorios  en un espacio matemático) . Este  proceso sobre el tiempo puede simularse en una dimensión $d$ comenzando en la posición $0$ moviendo la partícula una distancia aleatoria  $\Delta d$ respecto a su posición previa durante cierto intervalo de tiempo $\Delta t$. Donde la distancia   $\Delta d$ es muestreada de una distribución normal con media $0$ y variancia $\Delta t$. El muestreo de $\Delta d$ de la distribución normal se expresa denota como 
        """)
        st.latex(r"""\Delta d \sim \mathcal{N}""")
        st.write("""
        La figura siguiente muestra una simulación de $5$ trayectorias de un movimiento browniano (también conocidas como realizaciones).""")
        
        MBrowniano = Image.open('Movimiento_Browniano.png')
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
        
        
    
    VO_U_CODE = st.button("Visualizar/Ocultar Código Generación de u(x)",key="buttom VO_U_CODE")
    
    if VO_U_CODE:
        st.session_state.visualizar_ocultar_generacion_u_code = not st.session_state.visualizar_ocultar_generacion_u_code
        
    if st.session_state.visualizar_ocultar_generacion_u_code:
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
    
    #agree = st.checkbox('I agree')
    
    #st.write(agree)
    
    TEST_U_CODE = st.button("Probar Código de  Generación de u(x)",key="buttom TEST_U_CODE")
    
    if TEST_U_CODE:
        st.session_state.test_u_gen_code = not st.session_state.test_u_gen_code
    
    if st.session_state.test_u_gen_code:
        
        
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
            
            
            
        cal_procesos_gaussianos = st.button(label="GENERAR PROCESOS GAUSSIANOS",  key="Cal_PG")
        
        
        
        if cal_procesos_gaussianos:
            mean = np.ones(nb_of_samples)*media
            st.session_state.ys = np.random.multivariate_normal(mean,cov=COV,size=number_of_functions)
            fig2, ax2  = plt.subplots(figsize=(6, 4))
            for i in range(number_of_functions):
                ax2.plot(X, st.session_state.ys[i], linestyle='-', marker='o', markersize=3)
            ax2.set_xlabel('$x$', fontsize=13)
            ax2.set_ylabel('$u(x)$', fontsize=13)
            ax2.set_title('Procesos gaussianos generados')
            ax2.grid(True)
            st.pyplot(fig2)
            cal_procesos_gaussianos = False
            
            #st.write(type(ys))
            #st.write(np.shape(ys))
        
    st.write("""
    - SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL
    """)
    SOL_EQ_DIFF = st.button("Visualizar/Ocultar Solución Ecuación Diferencial",key="buttom SOL_EQ_DIFF")
        
    if SOL_EQ_DIFF:
        st.session_state.visualizar_ocultar_solucion_eq_diferencial = not st.session_state.visualizar_ocultar_solucion_eq_diferencial
            
    if st.session_state.visualizar_ocultar_solucion_eq_diferencial:
        u_default = 0.0
        if st.session_state.ys.all() == u_default:
            st.subheader("ANTES DE CONTINUAR NECESITA LAS FUNCIONES u(x)")
            st.subheader("Paso 1: --> Boton Probar Código de Generación de u(x)")
            st.subheader("Paso 2: --> Boton GENERAR PROCESOS GAUSSIANOS")
        else:
            u_options = []
            u_num_options = []
            u_num = np.linspace(0,number_of_functions,number_of_functions+1)
            for j in range(number_of_functions):
                u_options.append("u_"+str(int(j) ) +"(x)")
                #u_num_options.append(str(int(j) ))
            
            #st.write(u_options)
            #st.write(u_num_options)
            u_seleccionada  = st.multiselect(label="Seleccionar u(x) ", options=u_options)
            
            if not u_seleccionada:
                st.subheader("SELECCIONE AL MENOS UNA FUNCIÓN u(x)")
            else:
                u_index = []
                for k in u_seleccionada:
                    u_index.append(u_options.index(k) )
                #st.write(u_index)    
                fig3, ax3  = plt.subplots(figsize=(6, 4))
                for i in u_index:
                    ax3.plot(X, st.session_state.ys[i], label=u_options[i], linestyle='-', marker='o', markersize=3)
                ax3.set_xlabel('$x$', fontsize=13)
                ax3.set_ylabel('$u(x)$', fontsize=13)
                ax3.set_title('Procesos gaussianos generados')
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)
                
                op_seleccionado = st.multiselect(label="Selección del operador ", options=["ODE LINEAL","ODE NO LINEAL", "SISTEMA ODE NO LINEAL","PDE NO LINEAL"])
                
                
                if "ODE LINEAL" in op_seleccionado:
                    
                    col1, col2 = st.columns([1.5,2])
                    with col1:
                        s0_ode_lineal = st.slider('Selección de s(0)', 0.0, 1.0)
                        st.write("s(0) = ", s0_ode_lineal)
                    with col2:
                        st.latex(r'''\frac{ds(x)}{dx} = u(x) \quad x = [0,1]''')
                    
                    
                    sol_ode_linear_code = st.checkbox(label="EJECUTAR CALCULO NUMÉRICO", value=False, key="ODE_LINEAR_Sol")
                    
                    if sol_ode_linear_code:                        
                        interval = [0,1]
                        full_output = np.zeros(shape = (len(u_index) , np.shape(st.session_state.ys)[1] ) )
                        counter = 0
                        for i in u_index:
                            X = np.squeeze(X)
                            U = st.session_state.ys[i]
                            u = interp1d(X, U, kind='cubic')
                            model= lambda x, y :  u(x)
                            full_output[counter] = solve_ivp(model, interval, [s0_ode_lineal], method='RK45',t_eval=X,rtol = 1e-5).y
                            counter += 1
                            #sol = solve_ivp(model, interval, [s0_ode_lineal], method='RK45',t_eval=X,rtol = 1e-5)
                            #st.write(sol.t)
                            #st.write(sol.y)
                        fig4, ax4  = plt.subplots(figsize=(6, 4))
                        new_counter = 0
                        for solution in full_output:
                            ax4.plot(X, solution, label="s(x)--> "+u_options[u_index[new_counter]] , linestyle='-', marker='o', markersize=3)
                            new_counter +=1
                        ax4.set_xlabel('$x$', fontsize=13)
                        ax4.set_ylabel('$s(x)$', fontsize=13)
                        ax4.set_title('Soluciones')
                        ax4.grid(True)
                        ax4.legend()
                        st.pyplot(fig4)
                        #
                    
                    code_b = """
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import  solve_ivp
import matplotlib.pyplot as plt

###### DESCRIPCIÓN DEL PROBLEMA #############
#   
# Aqui queremos resolver la equación diferencial numéricamente
#
#                   ds(x)/dx = u(x)
#
# pero u(x) se presenta en forma de un arreglo de números
# pero la rutina numérica no acepta ese tipo de entrada
# por eso se define una función que da como salida los valores 
# interpolados de u(x) que es una entrada que la rutina numérica 
# si acepta, pero implica que es necesario verificar que el número  
# de puntos del arreglo de u(x) y el método de interpolación sean 
# adecuados para que la función interpolada sea una representación 
# precisa de u(x)

#
# Como este código es demostrativo vamos a utiliar una función u(x) 
# definida de forma simple u(x) = sin(x)
#

######################################################
#####   PASO 1: CREACIÓN DE FUNCIÓN INTERPOLABLE    ##  
######################################################

N = 512
X = np.linspace(0,2,N)              # El dominio de la función interpolable  
U = np.cos(X)                       # tiene que ser mayor que el dominio del
u = interp1d(X, U, kind='cubic')    # cálculo numérico


###########################################################
#####   PASO 2: DEFINICIÓN DE LA ECUACIÓN DIFERENCIAL    ##  
###########################################################

dy_dx = lambda x, y :  u(x)  # dy/dx = u(x)   # ODE

#########################################################
#####   PASO 3: SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL    ##  
#########################################################

interval = [0,1]   # Dominio de integración numérica

y_0 = 0.0            # Condición inicial  

x = np.linspace(0,1,N) # Puntos de evalación de la solución de la ODE

y_x = solve_ivp(dy_dx , interval, [y_0], method='RK45',t_eval=x,rtol = 1e-5).y

new_x = np.linspace(0,1,10)

s_x = np.sen(new_x)

plt.figure(figsize=(6, 4))
plt.plot(x, y_x, linestyle='-')
plt.plot(new_x, s_x, linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y(x)$', fontsize=13)
plt.title(('Solución numérica \\n solución analítica '))
plt.xlim([0, 1])
plt.show()


                    """
                    
                    
                    ver_sol_ode_linear_code = st.checkbox(label="Visualizar codigo", value=False, key="VC_ODE_LINEAR_Sol")
                    
                    if ver_sol_ode_linear_code:
                        st.code(code_b, language='python')
                    
                    
        
    
    st.write("""
    - Agrupar los datos de salida en una lista
    """)
    
    VO_DS = st.button("Visualizar/Ocultar Datos de salida",key="buttom VO_DS")
    
    if VO_DS:
        st.session_state.visualizar_ocultar_ode_lineal_codigo = not st.session_state.visualizar_ocultar_ode_lineal_codigo
        
    if st.session_state.visualizar_ocultar_ode_lineal_codigo:
        
        st.write(" DEFINICIÓN DE FUNCIÓN DE GENERACIÓN DE DATOS DE ENTRENAMIENTO")
        
        st.write("Parámetros de entrada de la función ")
        
        st.write(" - key -->  $\quad$ semilla para crear las posiciones de evaluación de la solución de forma aleatória")
        
        st.write(" - m --> $\quad$ número de puntos de x donde la función u(x) va a ser muestreada")
        
        st.write(" - P --> $\quad$ número de puntos va a ser calculada la solución de la ecuación diferencial")
        
        st.write(" - s_0 --> $\quad$ valor de la condición inicial ")
        
        st.write("Formato de salida de la función ")
        
        Data_Format = Image.open('DeepONet_Data.png')
        st.image(Data_Format)
        
        st.write("""
         en el diagrama  faltan dos una que corresponde a valores de condición inicial para cada uno de los cálculos y la otra que corresponde al valor de la función u en el punto de evaluación del operador
                 """)
        
        code_gen_single_data = """
import numpy as np

import scipy
from scipy.interpolate import interp1d
from scipy.integrate import  solve_ivp

import random

################################################################
##     DEFINICIÓN DEL NÚCLEO DE LA FUNCIÓN DE CORRELACION     ##
################################################################

def exponentiated_quadratic(xa, xb,sigma):
    #Exponentiated quadratic  with σ=sigma
    # L2 distance (Squared Euclidian)
    sq_norm = -(1/(2*sigma)) * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def generate_one_training_data(key ,m = 100, P = 1, s_0):    
    
    ######################################################
    ##    DEFINICIÓN DE LA FUNCION DE CORRELACIÓN       ##
    ##    NECESARIA PARA GENERAR EL PROCESO GAUSSIANO   ##
    ######################################################   
    
    nb_of_samples = 1024  # Número de puntos donde se va a evaluar
                          # el proceso estocástico

    # Muestras de variables independientes 
    X = np.expand_dims(np.linspace(0, 2, nb_of_samples), 1)
    
    sigma = 0.2                                 #
    COV = exponentiated_quadratic(X, X, sigma)  # Función de correlación
    
    ###########################################
    ##    GENERACIÓN DEL  PROCESO GAUSSIANO  ##
    ###########################################
    
    number_of_functions = 1  # Número de funciones a muestrear
                             # (número de procesos a generar)
                             
    mean = np.zeros(nb_of_samples) # Definición de la media (igual a cero)
    U = np.random.multivariate_normal(mean,cov=COV,size=number_of_functions) # generación de los procesos gaussianos
    
    ########################################################
    ##    DEFINICIÓN DE u(x) COMO FUNCIÓN INTERPOLABLE    ##
    ########################################################
    
    x = np.linspace(0,2,N)              # El dominio de la función interpolable  
    u = interp1d(x, U, kind='cubic')    # cálculo numérico
    z = np.linspace(0,1,m)
    u_output = np.tile(u(z),(P,1))      # Función evaluada en 
                                        # m puntos de prueba
                                        
    
    
    
    #################################################
    ##    DEFINICIÓN DE LA ECUACIÓN DIFERENCIAL    ##
    #################################################
    
    dy_dx = lambda x, y :  u(x)  # dy/dx = u(x)   # ODE
    
    ########################################################
    ####   DEFINICIÓN DE PUNTOS ALEATORIOS EVALUACIÓN   ####
    ####   DE LA SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL    ####
    ########################################################
    
    np.random.seed(seed=key)
    x_eval = np.random.rand(P,)
    
    ##############################################
    ##    SOLUCIÓN DE LA ECUACIÓN DIFERENCIAL   ##
    ##############################################
    
    interval = [0,1]   # Dominio de integración numérica
    y_x = solve_ivp(dy_dx , interval, [s_0], method='RK45',t_eval=x_eval,rtol = 1e-5).y
    
    
    return u_output, x_eval, y_x
        """
        
        ver_gen_single_data_code = st.checkbox(label="Visualizar codigo", value=False, key="VC_SINGLE_DATA_CODE")
        
        if ver_gen_single_data_code:
            st.code(code_gen_single_data, language='python')
        
        st.write("Creación de un arreglo")
        
        st.write("Mostrar resultado del cálculo para un punto aleatorio")
        
        st.write("Mostrar resultado del cálculo para N puntos específicos")
        
        st.write("Mostar resultado del cálculo para N puntos aleatorios")
        
        
        st.write("""
        Descripción de la agrupación de datos
        """)
        
        