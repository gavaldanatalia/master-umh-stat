import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, RocCurveDisplay, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
# pip install imbalanced-learn
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours
import matplotlib.pyplot as plt

# Cargar datos desde el archivo Excel
path = '/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/datos/'
df = pd.read_excel(path+'clasificacion_knn.xlsx')

# Separar variables independientes (X) y dependiente (y)
X = df[['X1', 'X2']]  # Variables independientes
y = df['Y']           # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parámetros a buscar para kNN: valores de k
param_grid = {'n_neighbors': np.arange(1, 32, 2)}

# Definir kNN y usar GridSearchCV para encontrar el mejor valor de k
modelo = KNeighborsClassifier()
# La función grid_search obtiene el mejor $k$ mediante validación cruzada 4-fold.
grid_search = GridSearchCV(modelo, param_grid, cv=4, scoring='roc_auc') # Maximizar el AUC de la curva ROC
# grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='accuracy')  # Maximizar la proporción de acierto
# grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='precision') # Minimizar falsos positivos
# grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='recall') # Minimizar falsos negativos
# grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='f1') # Balance entre precision y recall
# grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='roc_auc') # Maximizar el AUC de la curva ROC
grid_search.fit(X_train, y_train)

# Mejor valor de k encontrado
k = grid_search.best_params_['n_neighbors']
print(f"Mejor valor de k: {k}")

# Entrenar el modelo kNN con el mejor valor de k
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

# Guardar en un dataframe que exportemos a Excel:
y_pred_df = pd.DataFrame(y_pred, columns=['Predicción'])
y_prob_df = pd.DataFrame(y_prob, columns=['Probabilidad'])

df_resultado = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pred_df, y_prob_df], axis=1)
# df_resultado = pd.concat([X_test_df, y_test_df, y_pred_df, y_prob_df], axis=1)
df_resultado.to_excel('prueba1_knn.xlsx', index=False)


# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {roc_auc}")

RocCurveDisplay.from_estimator(modelo, X_test, y_test)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title(f'Curva ROC (AUC = {roc_auc:.2f})')
plt.show()

# Informe de clasificación y exactitud
print("\nInforme de clasificación:\n", classification_report(y_test, y_pred))
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))

# Validación cruzada 4-fold con kNN 
cv = StratifiedKFold(n_splits=4)
cv_scores = cross_val_score(modelo, X_train, y_train, cv=4, scoring='roc_auc')
print(f"Validación cruzada (4-fold) AUC: {np.mean(cv_scores)}")

# === Selección de prototipos usando CNN ===
cnn = CondensedNearestNeighbour(n_neighbors=k)
X_cnn, y_cnn = cnn.fit_resample(X_train, y_train)
df_prototipos = pd.concat([X_cnn, y_cnn], axis=1)
df_prototipos.to_excel('prototiposCNN.xlsx', index=False)

# Entrenar kNN con los prototipos seleccionados por CNN
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_cnn, y_cnn)

# Predicciones con prototipos CNN
y_pred = modelo.predict(X_test)
print("\nInforme de clasificación usando CNN:\n", classification_report(y_test, y_pred))
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))

# === Selección de prototipos usando ENN ===
enn = EditedNearestNeighbours(n_neighbors=3)
X_enn, y_enn = enn.fit_resample(X_train, y_train)
df_prototipos = pd.concat([X_enn, y_enn], axis=1)
df_prototipos.to_excel('prototiposENN.xlsx', index=False)

# Entrenar kNN con los prototipos seleccionados por ENN
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_enn, y_enn)

# Predicciones con prototipos ENN
y_pred = modelo.predict(X_test)
print("\nInforme de clasificación usando ENN:\n", classification_report(y_test, y_pred))
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))

###################################################################################################################
#### REGRESION con kNN ####
###################################################################################################################

df = pd.read_excel('regresion.xlsx')

# Definir las variables predictoras y la variable objetivo
X = df[['X1', 'X2', 'X3']]
y = df['Y']

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el rango de valores para k
param_grid = {'n_neighbors': range(1, 32, 2)}  # Probar k de 1 a 30

# Crear el modelo de regresión kNN
modelo = KNeighborsRegressor()

# Realizar la búsqueda de k usando validación cruzada
grid_search = GridSearchCV(modelo, param_grid, cv=4, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Obtener el mejor valor de k
k = grid_search.best_params_['n_neighbors']
print(f'El mejor valor de k: {k}')

# Ajustar el modelo con el mejor k
modelo_knn_best = KNeighborsRegressor(n_neighbors=k)
modelo_knn_best.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = modelo_knn_best.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir las métricas de evaluación
print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Coeficiente de determinación (R²): {r2:.2f}')

# Visualizar las predicciones frente a los valores reales
# Visualizar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))

# Graficar los valores observados en azul
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores observados', alpha=0.7)

# Graficar los valores predichos en rojo
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Valores predichos', alpha=0.7)

# Configuración de los ejes
plt.xlabel('Índice de observación')
plt.ylabel('Valores')
plt.title('Valores observados vs Valores predichos')
plt.legend()
plt.show()
