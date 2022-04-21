def linear_antiderivative_op_description(st, **state):
    st.header("Linear Antiderivative Operator Description")

    st.write("Concepto de operador")

    st.write("Descripción del operador antiderivada lineal")

    st.write("""
    
    a) ODE LINEAL
    """)
    st.latex(r'''
    \frac{ds(x)}{dx} = u(x) \quad s(0) = [a] \quad a \in \mathbb{R}
    ''')

    st.write("Concepto de operadores múltiples")
    
    st.write("Descripción de operadores múltiples antiderivada")

    st.write("Ejemplo de operadores múltiples a ser representado por la MONNet")

    st.write("""
    El operador que mapea del espacio de funciones u(x) al espacio de funciones s(x) cuando 
    
    a) ODE LINEAL
    """)
    st.latex(r'''
    \frac{ds(x)}{dx} = u(x) \quad s(0) = [0,1]
    ''')
    