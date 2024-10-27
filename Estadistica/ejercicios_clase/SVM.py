# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carga y Preparación de los Datos
# Leer los datos desde el archivo clasificacion.xlsx
path = '/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/datos/'
df = pd.read_excel(path+'clasificacion_svm.xlsx')

# Mostrar las primeras filas del conjunto de datos
print(df.head())

# Definir las variables independientes y dependiente
X = df[["X1", "X2"]] 
Y = df["Y"]

# 2. Obtención del parámetro C del SVM
# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Definir los valores de C a probar
C_pruebas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Realizar un ajuste de modelo SVM con validación cruzada para encontrar el mejor C
grid_parametros = {'C': C_pruebas}

#grid_parametros = {
#    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Diferentes kernels a probar
#    'degree': [2, 3, 4]  # Solo se aplica para el kernel polinómico
#}

busqueda = GridSearchCV(SVC(kernel='linear'), grid_parametros, cv=4)  # ¡Ojo! Linear utiliza la distancia Euclídea
# Directamente con sci-kit learn no hay manera de utilizar la distancia de Manhattan
busqueda.fit(X_train, y_train)

# Mostrar el mejor parámetro C
mejor_C = busqueda.best_params_['C']
print(f"Mejor valor de C: {mejor_C}")
# mejores_parametros = busqueda.best_params_

# 3. Evaluación del modelo
# Entrenar el modelo final con el mejor valor de C
modelo = SVC(C=mejor_C, kernel='linear')
#modelo = SVC(kernel='poly', degree=3) # Kernel polinómico
#modelo = SVC(kernel='rbf') # Kernel Radial Basis Function. Es uno de los más utilizados
#svm_model = SVC(kernel='sigmoid') # Semejante a la regresión logística y las redes neuronales

modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Evaluar el modelo
print("Informe de clasificación:")
print(classification_report(y_test, predicciones))

# 4. Obtención de los pesos del hiperplano (sólo se puede realizar con SVM lineal)
pesos = modelo.coef_[0]
intercepto = modelo.intercept_
print("Intercepto y pesos del hiperplano:")
print(intercepto, pesos)

# 6. SVR (Support Vector Regression)
# Leer los datos para regresión desde el archivo regresion.xlsx
df = pd.read_excel(path+"regresion.xlsx")

# Definir las variables independientes y la variable dependiente
X = df[["X1", "X2", "X3"]]  
Y = df["Y"]

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Ajustar el modelo SVR para distintos valores de C
C_pruebas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
epsilon_pruebas = [0.001, 0.01, 0.1, 1, 10]
grid_parametros = {'C': C_pruebas, 'tol': epsilon_pruebas}
busqueda = GridSearchCV(SVR(kernel='linear'), grid_parametros, cv=4) # ¡Ojo! Linear utiliza la distancia Euclídea
# Directamente con sci-kit learn no hay manera de utilizar la distancia de Manhattan
busqueda.fit(X_train, y_train)

# Mostrar el mejor parámetro C
mejor_C = busqueda.best_params_['C']

print(f"Mejor valor de C: {mejor_C}")

# Entrenar el modelo SVR con el mejor C
modelo = SVR(C=mejor_C, kernel='linear')
modelo.fit(X_train, y_train)

# Realizar predicciones
predicciones = modelo.predict(X_test)

# Calcular el error cuadrático medio (MSE) y el R^2
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)
print(f"MSE: {mse}")
print(f"R^2: {r2}")

# Gráfico de valores observados vs predichos
plt.scatter(y_test, predicciones, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title("Valores observados vs predichos")
plt.xlabel("Valores observados")
plt.ylabel("Valores predichos")
plt.show()

# 6. Obtención de los pesos del hiperplano en SVR  (sólo tiene sentido con Kernel lineal)
pesos = modelo.coef_[0]
intercepto = modelo.intercept_
print("Pesos del hiperplano (SVR):", pesos, intercepto)
