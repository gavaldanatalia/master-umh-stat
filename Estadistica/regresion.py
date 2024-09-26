import pandas as pd
import numpy as np

# Leer el archivo Excel en un DataFrame
path = '/Users/jjmilla/Repositorios/master-umh/Estadistica/'
df = pd.read_excel(path+'regresion.xlsx')

# Mostrar las primeras filas del DataFrame
print("--- Datos:")
print(df.head())

# Definir las variables independientes (X) y la variable dependiente (Y)
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

#############################################
# Obtener 'a pelo' los beta de la regresión #
#############################################
X = np.array(X)
Y = np.array(Y)
# Agregamos la columna de 1s a X para el intercepto
# Si no se agregase no habria intercepto (b0) en la regresion
X = np.hstack([np.ones((X.shape[0], 1)), X])
# Formula MCO: (X’X)^-1 X’Y
X_t = X.T
beta = np.linalg.inv(X_t @ X) @ (X_t @ Y)
print('---')
print(f"Coeficientes:\n{beta}")

# Calculo las predicciones
Y_pred = X @ beta

# Las añado a df y lo grabo en Excel
df['Predicciones']=Y_pred
df.to_excel('predicciones.xlsx')

# Cálculo de errores
errores = Y - Y_pred

# Cálculo de SSE (Suma de los errores al cuadrado)
SSE = np.sum(errores ** 2)

# Cálculo de SST (Suma total de cuadrados)
Y_mean = np.mean(Y)
SST = np.sum((Y - Y_mean) ** 2)

# Cálculo de R^2
R2 = 1 - (SSE / SST)

# Cálculo de R^2 ajustado
n = X.shape[0]  # Número de observaciones
p = X.shape[1]-1  # Número de predictores (variables independientes excluyendo el intercepto)

R2_ajustado = 1 - (1 - R2) * (n - 1) / (n - p - 1)


print(f"R^2: {R2}")
print(f"R^2 ajustado: {R2_ajustado}")
print("-------------------------")
#############################################
# Usando la librería scikit-learn #
#############################################

from sklearn.linear_model import LinearRegression
X = df[['X1', 'X2', 'X3']]
Y = df['Y']

X = np.array(X)
Y = np.array(Y)

# Crear el modelo de reg. lineal
modelo = LinearRegression(fit_intercept=True)
# Ajustar el modelo a los datos
modelo.fit(X, Y)
# Coeficientes del modelo
beta = np.concatenate(([modelo.intercept_], modelo.coef_))
print(f"Coeficientes según sklearn: {modelo.coef_}") 
print(f"Intercepto según sklearn: {modelo.intercept_}")
print(f"Coeficientes finales según sklearn:\n{beta}")

############################
# Obtención de los errores #
############################
Y_pred = modelo.predict(X)
errores = Y - Y_pred
R2 = modelo.score(X, Y)
n = X.shape[0]  # Número de observaciones
p = X.shape[1]  # Número de predictores (variables independientes)

R2_ajustado = 1 - (1 - R2) * (n - 1) / (n - p - 1)
print(f"R^2 según sklearn: {R2}")
print(f"R^2 ajustado según sklearn: {R2_ajustado}")
print("-------------------------")
