# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Carga y Preparación de los Datos
# Cargar el archivo Excel
df = pd.read_excel('clasificacion.xlsx')

# Visualizar las primeras filas del conjunto de datos
print(df.head())

# Definir las variables independientes y dependientes
X = df[['X1', 'X2']]  
Y = df['Y']

# 2. Obtención del modelo
# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Crear el modelo de árbol de RandomForest
modelo = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
modelo.fit(X_train, Y_train)

# 3. Evaluación del modelo
# Obtener pronósticos
predicciones = modelo.predict(X_test)

# Obtener la tabla de confusión
conf_matrix = confusion_matrix(Y_test, predicciones)
print("Tabla de Confusión:\n", conf_matrix)

# Obtener el informe de clasificación
informe = classification_report(Y_test, predicciones)
print("Informe de Clasificación:\n", informe)

# 4. Interpretación
# Obtener la importancia de las variables
importancia_variables = modelo.feature_importances_
print("Importancia de las Variables:\n", importancia_variables)

# Graficar la importancia de las variables
features = X.columns
indices = np.argsort(importancia_variables)

plt.figure(figsize=(8, 6))
plt.title("Importancia de las Variables")
plt.barh(range(len(indices)), importancia_variables[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Importancia Relativa")
plt.show()

# 5. Probar distintos parámetros
modelo = RandomForestClassifier(n_estimators=200, max_features=2, min_samples_split=4, min_samples_leaf=2, random_state=42)
modelo.fit(X_train, Y_train)
predicciones = modelo.predict(X_test)
conf_matrix = confusion_matrix(Y_test, predicciones)
print("Tabla de Confusión:\n", conf_matrix)
informe = classification_report(Y_test, predicciones)
print("Informe de Clasificación:\n", informe)

# 6. Árboles de decisión aplicados a regresión
# Cargar el archivo Excel para la regresión
df = pd.read_excel('regresion.xlsx')

# Definir las variables independientes y dependiente
X_reg = df[['X1', 'X2', 'X3']] 
Y_reg = df['Y']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=0.3, random_state=42)

# Inicializar y entrenar el modelo de Random Forest para regresión
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, Y_train)

# Hacer predicciones sobre el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular el error cuadrático medio y el R^2
mse = mean_squared_error(Y_test, predicciones)
r2 = r2_score(Y_test, predicciones)
print(f'Error Cuadrático Medio: {mse}')
print(f'R^2: {r2}')

# Graficar valores observados vs predichos
plt.figure(figsize=(10, 6))

plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Valores Observados')
plt.scatter(range(len(Y_test)), predicciones, color='orange', label='Valores Predichos')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valores')
plt.title('Valores Observados vs Predichos')

plt.legend()
plt.show()

