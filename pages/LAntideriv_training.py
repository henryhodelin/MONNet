import pandas as pd
import numpy as np


def linear_antiderivative_training(st, **state):
    
    df = pd.DataFrame(
    np.random.randn(10, 5),
    columns=('col %d' % i for i in range(5)))

    #st.table(df)

    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)



    st.header("Linear Antiderivative  Training ")

    st.write(""" 
    CARACTERISTICAS DE LA COMPUTADORA
    -PROCESADOR

    -MEMORIA RAM

    -SISTEMA OPERATIVO 

    
    """)

    with st.expander("Single Antiderivative Operator Training"):
        st.write("""
        DETALLES DEL ENTENAMIENTO
        - TIEMPO QUE TARDO

        - GRAFICO DE LA EVOLUCIÓN DE LA FUNCIÓN DE PERDIDA

        - EJEMPLOS DE RESULTADOS


        """ )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('Jupyter notebooks')        
            st.download_button(
            label="Download Single antiderivative operator",
            data=csv,
            file_name='large_df.csv',
            mime='text/csv',
            )
        with col2:
            st.subheader(' .py')        
            st.download_button(
                label="Download Single antiderivative operator .py",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
                )
        with col3:
            st.subheader('PDF')        
            st.download_button(
                label="Download Single antiderivative operator PDF",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
                )
        

    with st.expander("Multiple Antiderivative Operator Training"):
        st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('Jupyter notebooks')        
            st.download_button(
                label="Download Multiple antiderivative operator",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
                )
        with col2:
            st.subheader(' .py')        
            st.download_button(
                label="Download Multiple antiderivative operator .py",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
                )
        with col3:
            st.subheader('PDF')        
            st.download_button(
                label="Download Multiple antiderivative operator PDF",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
                )

    
    
    
    
    
    




        

    