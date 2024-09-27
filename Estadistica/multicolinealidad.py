import numpy as np
import pandas as pd


# Leer el archivo Excel en un DataFrame
path = '/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/'
df = pd.read_excel(path+'multicolinealidad.xlsx')

# Mostrar las primeras filas del DataFrame
print("--- Datos:")
print(df.head())

X= df[['X1', 'X2', 'X3']]
correlaciones = X.corr()

# Mostrar la matriz de correlación
print(correlaciones)

X= df[['X2', 'X3']]
correlaciones = X.corr()

# Mostrar la matriz de correlación
print(correlaciones)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calcular el VIF para cada variable independiente
X = df[['X1', 'X2', 'X3']]
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


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



# Gráfica de valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)  # Línea de ajuste perfecto
plt.title('Valores reales vs Valores predichos')
plt.xlabel('Valores reales de Y')
plt.ylabel('Valores predichos de Y')
plt.grid(True)
plt.show()

#############################################################################
## Regresión con variables no correladas ####
#############################################################################

# Definir las variables independientes (X) y la variable dependiente (Y)
X = df[[ 'X2', 'X3']]
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


# Gráfica de valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(Y, Y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)  # Línea de ajuste perfecto
plt.title('Valores reales vs Valores predichos')
plt.xlabel('Valores reales de Y')
plt.ylabel('Valores predichos de Y')
plt.grid(True)
plt.show()
