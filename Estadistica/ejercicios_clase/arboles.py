import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

# 1. Carga y Preparación de los Datos
# Cargar el archivo Excel
df = pd.read_excel('clasificacion.xlsx')

# Visualizar las primeras filas del conjunto de datos
print(df.head())

# Definir las variables independientes y dependientes
X = df[['X1', 'X2']]  
y = df['Y'] 

# 2. Obtención del modelo
# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de árbol de decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 3. Evaluación del modelo
# Obtener pronósticos
predicciones = modelo.predict(X_test)

# Obtener la tabla de confusión
conf_matrix = confusion_matrix(y_test, predicciones)
print("Tabla de Confusión:")
print(conf_matrix)

# Obtener el informe de clasificación
informe = classification_report(y_test, predicciones)
print("Informe de Clasificación:")
print(informe)

# 4. Interpretación
# Graficar el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(modelo, 
           filled=True, 
           feature_names=['X1', 'X2'],  # Cambia según tus características
           class_names=[str(cls) for cls in modelo.classes_])  # Convierte a cadenas
plt.title('Árbol de Decisión')
plt.show()

# 5. Probar distintos parámetros
modelo = DecisionTreeClassifier(random_state=42, max_depth=2)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
conf_matrix = confusion_matrix(y_test, predicciones)
print("Tabla de Confusión:")
print(conf_matrix)
informe = classification_report(y_test, predicciones)
print("Informe de Clasificación:")
print(informe)

plt.figure(figsize=(12, 8))
plot_tree(modelo, 
           filled=True, 
           feature_names=['X1', 'X2'],  # Cambia según tus características
           class_names=[str(cls) for cls in modelo.classes_])  # Convierte a cadenas
plt.title('Árbol de Decisión')
plt.show()

# Ejemplo de grid de parámetros de búsqueda
#grid_parametros= {
#    'criterion': ['gini', 'entropy'],
#    'max_depth': [None, 5, 10, 15],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4]
#}


# 6. Árboles de decisión aplicados a regresión
# Cargar el archivo Excel para la regresión
df = pd.read_excel('regresion.xlsx')

# Definir variables independientes y dependientes para regresión
X = df[['X1', 'X2', 'X3']]  # Cambia 'X1', 'X2', 'X3' por los nombres reales de tus columnas
y = df['Y']  # Cambia 'Y' por el nombre real de tu variable dependiente

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de árbol de decisión para regresión
modelo = DecisionTreeRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones
predicciones = modelo.predict(X_test)

# Calcular el error cuadrático medio y R^2
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f'Error Cuadrático Medio: {mse}')
print(f'R^2: {r2}')

# Graficar valores observados frente a valores predichos
plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Observados')
plt.scatter(range(len(y_test)), predicciones, color='orange', label='Valores Predichos')
plt.xlabel('Índice de la muestra')
plt.ylabel('Valores')
plt.title('Valores Observados vs Predichos')

plt.legend()
plt.show()
