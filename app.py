import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import load
from joblib import dump
from PIL import Image

#sns.set_theme(style="darkgrid")


st.set_page_config(
   page_title="🤖 Camilo Franco",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

with st.container():
    col1, col2 = col1, col2 = st.columns([3,2],gap='small')
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.header("Demo Modelo Data Scoring  📈")
        st.header(":orange[SERMUTUAL]")
        st.markdown("Elaborado por: Camilo Franco Data Scientist")
        st.markdown('''- Github [@camigenius](https://github.com/camigenius)''')
    with col2:
        image = Image.open('Dashboard.jpg')
        st.image(image, caption='Foto Unplash,Autor: Pierre Bamin, https://unsplash.com/license')     


Data = "https://github.com/gastonstat/CreditScoring/raw/master/CreditScoring.csv"

df = pd.read_csv(Data)

st.markdown(''':red[Variables del Modelo una vez superada la Etapa EDA( Exploratory Data Analysis).]''')

st.markdown("""
- :green[Status-> Estado:]  si el cliente logró pagar el préstamo (1) o no (2)
- :green[Seniority-> Antigüedad:]  experiencia laboral en años
- :green[Home-> hogar:]  tipo de propiedad: alquiler (1), propietario (2) y otros
- :green[Time-> tiempo:]  período planeado para el préstamo (en meses)
- :green[Age-> edad:]  edad del cliente
- :green[MaritaL-> estado civil:]  soltero (1), casado (2) y otros
- :green[Records-> registros:]  si el cliente tiene algún registro previo: no (1), sí (2) (No está claro en la descripción del conjunto de datos qué tipo de registros hay en esta columna. Para los propósitos de este proyecto, podríamos asumir que se trata de registros en la base de datos del banco.)
- :green[Job-> trabajo:]  tipo de trabajo: tiempo completo (1), tiempo parcial (2) y otros
- :green[Expenses-> gastos:]  cuánto gasta el cliente por mes
- :green[Income-> ingresos:]  cuánto gana el cliente por mes
- :green[Assets-> activos:]  valor total de todos los activos del cliente
- :green[Debt-> deuda:]  cantidad de deuda de crédito
- :green[Amount-> monto:]  monto solicitado del préstamo
- :green[Price->precio:]  precio de un artículo que el cliente quiere comprar
""")


st.write("---")

st.table(df.head())

df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unk']


st.divider()
st.markdown(''':green[ok (1)      ----> Cliente SÍ logró Pagar Crédito .]''')
st.markdown(''':red[  default (2) ----> Cliente  NO logró Pagar Crédito .]''')
st.table(df.status.value_counts())

st.subheader('Modelos Árbol de Desición (DecisionTreeClassifier)')
st.subheader('AUC : :green[0.785]')

image1 = Image.open('RocArbolDecision.png')
st.image(image1, caption='Curva Roc Árbol de Decisión')

st.subheader('Modelo Bosques Aleatorios (RandomForestClassifier)')
st.subheader('AUC : :green[0.823]')

image2 = Image.open('AreasBajoCurvaBosuqesAleatorios.png')
st.image(image2, caption='Prueba con diferentes Números de Árboles ')

image3 = Image.open('AreasBajoCurvaBosuqesProfundidad.png')
st.image(image3, caption='Prueba con diferentes Números de Árboles y diferentes profundidades')

st.subheader('Comparación curva ROC entre los dos Modelos')
image4 = Image.open('ComparacionModelos.png')
st.image(image4, caption='Comparación Modelos Árboles de Desición y Bosques Aleatorios')



df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)

y_train_full = (df_train_full.status == 'default').values
y_test = (df_test.status == 'default').values

del df_train_full['status']
del df_test['status']

st.divider()

st.subheader('El set de datos se particiona en Entrenamiento, Validación y Prueba.')

st.write('Tamaño Total del Set de Datos : ',len(df))
st.write('Tamaño set de Entrenamiento : ',len(df_train))
st.write('Tamaño set de Validación : ',len(df_val))
st.write('Tamaño set de prueba : ',len(df_test))

dict_train_full = df_train_full.fillna(0).to_dict(orient='records')
dict_test = df_test.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train_full = dv.fit_transform(dict_train_full)
X_test = dv.transform(dict_test)

rf_final = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=1)
rf_final.fit(X_train_full, y_train_full)

y_pred_rf = rf_final.predict_proba(X_test)[:, 1]

code = '''
rf_final = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=1)
rf_final.fit(X_train_full, y_train_full)
'''


st.subheader('Entrenamos nuestro modelo con los mejores  parámetros que optimizan el modelo')
st.code(code, language='python')
st.write(':green[n_estimators = ]    Cantidad de arboles')
st.write(':green[max_depth = ]       Profundidad de cada arbol (niveles ó ramas)')
st.write(':green[min_samples_leaf =] Hojas ')



st.divider()



st.subheader(':blue[Con nuestro modelo ya entrenado  con los mejores Parámetros ahora podemos hacer un predicción!!🤖]')

st.subheader('Podemos elejir cualquiera de los 891 clientes que particionamos para hacer las pruebas')
number = st.number_input('Ingrese por favor el número de 1 al 891',step=1,min_value=1, max_value=891)
st.write('Usted ha seleccionado el cliente No ', number)

st.table(df_test[number - 1:number])

result = st.button("Clic Aquí para ejeutar predicción")

# st.table(df_test[0:23])
#st.table(y_pred_rf[0:23])

if result:
    single_pred_proba = rf_final.predict_proba(X_test)[number-1]
    single_pred_bin = rf_final.predict(dv.transform(dict_test[number-1]))
    st.write("La probabilidad de SÍ pagar el Crédito es ✅ ",single_pred_proba[0])
    st.write("La probabilidad de NO pagar el Crédito es ❌",single_pred_proba[1])
    if single_pred_bin:
        st.write("La predicción es ",single_pred_bin[0],"Se recomienda Negar el Crédito")
    else:
        st.write("La predicción es ",single_pred_bin[0],"Se recomienda Aprobar el Crédito")

# CREDITO 90 Y 23
