import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats  

###############
def evaluar_modelo(modelo, X, Y, X_test, Y_test, nombre="Regresión"):
    # Evalúa el modelo
    R2 = modelo.score(X, Y)
    
    # Imprimir las métricas
    print(f"{nombre}:")
    print(f"  - R^2 entrenamiento: {R2}")

    Y_predicho_test = modelo.predict(X_test)
    
    R2_test = r2_score(Y_test, Y_predicho_test)

    print(f"  - R^2 test: {R2_test}")
    print()

def validar_modelo(Y_observado, Y_pred, nombre="Regresión"):
    # Calcular los errores
    errores = Y_observado - Y_pred
    
    # Crear las gráficas
    plt.figure(figsize=(10, 5))
    
    # Histograma de los errores
    plt.subplot(1, 2, 1)
    plt.hist(errores, bins=20, color='red', alpha=0.7)
    plt.title('Histograma de los Errores -' + nombre)
    plt.xlabel('Errores')
    plt.ylabel('Frecuencia')

    # Gráfico Q-Q de los errores
    plt.subplot(1, 2, 2)
    stats.probplot(errores, dist="norm", plot=plt)
    plt.title('Gráfico Q-Q de los Errores -' + nombre)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()
#######

# Cargar el Boston Housing Dataset desde la URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# Procesar los datos (combinar las filas en pares y separar en datos y target)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Definir las columnas de las variables independientes basadas en el conjunto original
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Crear el DataFrame
# data = pd.read_csv('nombre_fichero.csv', sep="\t" header=None)
df = pd.DataFrame(data, columns=columns)
df['MEDV'] = target

# Exploración estadística de las variables
print(df.describe() )

# Revisar si hay valores nulos
nulos = df.isnull().sum() # Valores nulos por columna
total_nulos = nulos.sum()

if(total_nulos>0):
    print('ATENCION!!! Hay valores nulos: ' , total_nulos)
    registros_con_nulos = df[df.isnull().any(axis=1)]
    print('Registros con valores nulos: ', registros_con_nulos)
    print('Valores nulos según columna: ', nulos)
    # Plantearse si hay que eliminar registros
    # df = df.drop(index=1)
    # Plantearse si hay que eliminar columnas
    # df = df.drop(columns=['nombre'])
    input()


# Diagrama de correlación entre variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de Correlación entre Variables")
plt.show()

# Plantearse si según el diagrama de correlación habría que borrar variables:
# Separar en variables dependientes e independientes

Y = df['MEDV']
X = df.drop(columns=['MEDV'])


# Dividir los datos en entrenamiento y prueba
X_entrenamiento, X_test, Y_entrenamiento, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# El número 42 se ha popularizado como un valor "mágico" en la programación y en la cultura geek por su mención en el libro "La guía del autoestopista galáctico" de Douglas Adams, donde se describe como "la respuesta a la vida, el universo y todo lo demás".  Muchos desarrolladores lo utilizan como un número de referencia en ejemplos de código o de inicialización de semillas. En verdad, se podría utilizar cualquier otro número como semilla.

# Ajuste del Modelo de Regresión Lineal
modelo= LinearRegression()
modelo.fit(X_entrenamiento, Y_entrenamiento)

# Coeficientes e Intercepto
print(f'Coeficientes de la regresión lineal: {modelo.coef_}')
print(f'Intercepto: {modelo.intercept_}')
# Diagrama de los coeficientes
plt.bar(X.columns, modelo.coef_)
plt.title("Coeficientes del Modelo de Regresión Lineal")
plt.xlabel("Variables")
plt.ylabel("Coeficientes")
plt.xticks(rotation=90)
plt.show()

# Evaluación del modelo
Y_pred_entrenamiento = modelo.predict(X_entrenamiento)
Y_pred_test = modelo.predict(X_test)

# Evaluar el modelo de regresión lineal
evaluar_modelo(modelo, X_entrenamiento, Y_entrenamiento, X_test, Y_test, "Regresión Lineal")
validar_modelo(Y_entrenamiento, Y_pred_entrenamiento, "Regresión Lineal")

# Transformación de Variables
# Un ejemplo: X=X**2
errores= Y_pred_entrenamiento-Y_entrenamiento
plt.scatter(X_entrenamiento['RM'], errores, color="blue", alpha=0.5)
plt.title(f"Residuos vs {'RM'}")
plt.xlabel('RM')
plt.ylabel("Residuos")
plt.show()


X_entrenamiento['RM_cuadrado'] = X_entrenamiento['RM']**2  # Añadir nueva variable cuadrática de un predictor
X_test['RM_cuadrado'] = X_test['RM']**2

modelo2= LinearRegression()
modelo2.fit(X_entrenamiento, Y_entrenamiento)
print(f'Coeficientes del modelo transformado: {modelo.coef_}')
print(f'Intercepto: {modelo.intercept_}')


evaluar_modelo(modelo2, X_entrenamiento, Y_entrenamiento, X_test, Y_test, "Transformación")
validar_modelo(Y_entrenamiento, Y_pred_entrenamiento,"Transformación")

