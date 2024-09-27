import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

path = '/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/'
df = pd.read_excel(path+'heterocedasticidad_normalidad.xlsx')

########################################################################
## REGRESIÓN ##
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Definir las variables independientes (X) y la variable dependiente (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(X, Y)

# Coeficientes del modelo
print(f'Coeficientes: {modelo.coef_}')
print(f'Intercepto: {modelo.intercept_}')


# Obtener el R^2
r_2 = modelo.score(X, Y)
print(f"R^2: {r_2}")

# Calcular el R^2 ajustado
n = X.shape[0]  # Número de observaciones
p = X.shape[1]  # Número de predictores (variables independientes)

r_2_ajustado = 1 - (1 - r_2) * (n - 1) / (n - p - 1)
print(f"R^2 ajustado: {r_2_ajustado}")

# Hacer predicciones
Y_pred = modelo.predict(X)

# Mostrar las primeras predicciones y el valor real
df_predictions = pd.DataFrame({
    'Y_real': Y,
    'Y_pred': Y_pred,
    'Error': Y - Y_pred
})
print(df_predictions.head())

errores=  Y - Y_pred

# Gráficos de los errores para visualizar la mayor heterocedasticidad
for column in X.columns:
    plt.scatter(X[column], errores, color="blue", alpha=0.5)
    plt.title(f"Residuos vs {column}")
    plt.xlabel(column)
    plt.ylabel("Residuos")
    plt.show()

##################################################
# Varios gráficos en uno solo
# Definir el número de columnas (variables independientes) en el DataFrame X
num_columns = len(X.columns)

# Crear una figura con subplots
# fig, axs = plt.subplots(1, num_columns, figsize=(15, 5))  # 1 fila y num_columns columnas
num_rows = math.ceil(num_columns / 2)
fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))  # 2 columnas y calculamos el número de filas

# Aplanar el array de axs si tiene más de una fila, para poder iterar fácilmente
axs = axs.ravel()

# Iterar sobre todas las columnas de X y generar un subplot para cada una
for i, column in enumerate(X.columns):
    axs[i].scatter(X[column], errores, color="blue", alpha=0.5)
    axs[i].set_title(f"Residuos vs {column}")
    axs[i].set_xlabel(column)
    axs[i].set_ylabel("Residuos")

# Eliminar los subplots vacíos si hay un número impar de columnas
if num_columns % 2 != 0:
    fig.delaxes(axs[-1])  # Elimina el último subplot vacío si hay un número impar de gráficos

# Ajustar el diseño de la figura para evitar solapamientos
plt.tight_layout()

# Mostrar todos los subgráficos
plt.show()
###################################################
# Histograma de residuos no normales
plt.subplot(1, 2, 1)
plt.hist(errores, bins=10, color='red', alpha=0.7)
plt.title('Histograma de Residuos (No Normales)')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')

# Gráfico Q-Q de residuos no normales
plt.subplot(1, 2, 2)
stats.probplot(errores, dist="norm", plot=plt)
plt.title('Gráfico Q-Q de Residuos (No Normales)')

# Mostrar los gráficos
plt.tight_layout()
plt.show()
