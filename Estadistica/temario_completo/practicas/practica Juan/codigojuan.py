# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:43:03 2024

@author: Alfonso López
"""
# Importar bibliotecas necesarias
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error, r2_score, RocCurveDisplay
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,  cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

def cambiarBooleanos(X):
    for columna in X.columns:
        if X[columna].dtype == 'bool':
            X[columna] = X[columna].replace({True: 1, False: 0})

def eliminarVifAltos(X):
#Calcular vif de variables para el data frame
    while True:
        vif_data = pd.DataFrame()
        vif_data["variable"] = X.columns
        
        cambiarBooleanos(X)
        
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print(vif_data)
        # Si todas las VIF están por debajo de 10, salimos del bucle
        if vif_data['VIF'].max() <= 10:
            break

        # Eliminamos la variable con el VIF más alto
        mayor_vif = vif_data.loc[vif_data['VIF'].idxmax(), 'variable']
        X = X.drop(mayor_vif, axis=1)
        print(vif_data)
        
    #Devolvemos dataFrame
    return X

def regresionLogistica(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    
    # Ajustar el modelo de regresión logística
    modelo = LogisticRegression(max_iter=9000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Aumentar el umbral de 0.5 a 0.7
    y_prob = modelo.predict_proba(X_test)[:, 1]
    umbral = 0.12
    y_pred2 = (y_prob >= umbral).astype(int)  # Clasificar como 1 si la probabilidad es mayor o igual al umbral

    # Obtener los coeficientes del modelo
    coefficients = np.append(modelo.intercept_, modelo.coef_)  # Añadir intercepto a los coeficientes
    print("\nCoeficientes del modelo:")
    print(coefficients)

    # Tabla de confusión y otras métricas con sklearn
    conf_matrix = confusion_matrix(y_test, y_pred2)

    conf_matrix_df = pd.DataFrame(conf_matrix,
                                   index=['Real Negativo', 'Real Positivo'],  # Orden cambiado
                                   columns=['Predicción Negativa', 'Predicción Positiva'])  # Orden cambiado
    print("\nTabla de Confusión con sklearn:")
    print(conf_matrix_df)
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred2))
    print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred2))

    #y_prob = modelo.predict_proba(X_test)[:, 1]
    roc_auc=roc_auc_score(y_test, y_prob)


    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    print(thresholds)

    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')

    plt.legend(loc="lower right")
    plt.show()
    '''
    errores=  y_test - y_pred
    
    # Gráficos de los errores para visualizar la mayor heterocedasticidad
    for column in X.columns:
        plt.scatter(X_test[column], errores, color="blue", alpha=0.5)
        plt.title(f"Residuos vs {column}")
        plt.xlabel(column)
        plt.ylabel("Residuos")
        plt.show()
        
    
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
    '''
    
def kNN(X,y):
    # Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Parámetros a buscar para kNN: valores de k
    param_grid = {'n_neighbors': np.arange(1, 32, 2)} #crea un vector de k empezando por 1 y va subiendo de 2 en 2

    # Definir kNN y usar GridSearchCV para encontrar el mejor valor de k
    modelo = KNeighborsClassifier() #Defina el modelo que será utilizado
    # La función grid_search obtiene el mejor $k$ mediante validación cruzada 4-fold. (cv es el k del kfold)
    grid_search = GridSearchCV(modelo, param_grid, cv=10, scoring='precision')
    # grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='accuracy')  # Maximizar la proporción de acierto - es lo que el profesor suele utilizar
    # grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='precision') # Minimizar falsos positivos
    # grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='recall') # Minimizar falsos negativos
    # grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='f1') # Balance entre precision y recall
    # grid_search = GridSearchCV(knn, param_grid, cv=4, scoring='roc_auc') # Maximizar el AUC de la curva ROC
    grid_search.fit(X_train, y_train)

    # Mejor valor de k encontrado
    k = grid_search.best_params_['n_neighbors']
    print(f"Mejor valor de k: {k}")

    # Entrenar el modelo kNN con el mejor valor de k
    modelo = KNeighborsClassifier(n_neighbors=k) #la variable metric es la utilizada para definir la distancia, por defecto utiliza euclidiana. Podría utilizar manhattan
    # variable weight es la que se utilizaria para poderar las variables
    # p=1 utilizaria manhatan, p=2 euclidiana y p=0 calcula la distancia minima L0, P=INFINITO utiliza chebyshev
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
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
    
    # Validación cruzada 4-fold con kNN - aplica 4 fold en el modelo de test mezclado con training
    cv = StratifiedKFold(n_splits=4)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=4, scoring='roc_auc')
    print(f"Validación cruzada (4-fold) AUC: {np.mean(cv_scores)}")


    # === Selección de prototipos usando ENN ===
    enn = EditedNearestNeighbours(n_neighbors=3)
    X_enn, y_enn = enn.fit_resample(X_train, y_train)
    df_prototipos = pd.concat([X_enn, y_enn], axis=1)
    df_prototipos.to_excel('prototiposENN.xlsx', index=False)

    # Entrenar kNN con los prototipos seleccionados por ENN
    modelo = KNeighborsClassifier(n_neighbors=k, metric="minkowski")
    modelo.fit(X_enn, y_enn)

    # Predicciones con prototipos ENN
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    umbral = 0.25
    y_pred2 = (y_prob >= umbral).astype(int)
    print("\nInforme de clasificación usando ENN:\n", classification_report(y_test, y_pred2))
    print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred2))
    
    # Tabla de confusión y otras métricas con sklearn
    conf_matrix = confusion_matrix(y_test, y_pred2)

    conf_matrix_df = pd.DataFrame(conf_matrix,
                                       index=['Real Negativo', 'Real Positivo'],  # Orden cambiado
                                       columns=['Predicción Negativa', 'Predicción Positiva'])  # Orden cambiado
    print("\nTabla de Confusión con sklearn:")
    print(conf_matrix_df)

def SVM(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Suponiendo que tienes tus datos en X (características) e y (etiquetas)

    
    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(X_train)
    
    # Crear y entrenar el modelo SVM
    #model = SVC(kernel='poly')  # Puedes cambiar el kernel a 'rbf', 'poly', etc.
    #model.fit(X_train, y_train)
    
    # Hacer predicciones
    #y_pred = model.predict(X_test)

    grid_parametros = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Diferentes kernels a probar
        'degree': [2,3,4]  # Solo se aplica para el kernel polinómico #grado del kernel
    }
    print("Hola2")
    
    #Linear es la euclídea, interesaría probarlos y comparar
    busqueda = GridSearchCV(SVC(), grid_parametros, cv=2)  # Â¡Ojo! Linear utiliza la distancia EuclÃ­dea
    print(busqueda)
    
    # Directamente con sci-kit learn no hay manera de utilizar la distancia de Manhattan
    
    busqueda.fit(X_train, y_train)
    
    # Mostrar el mejor parÃ¡metro C
    mejor_C = busqueda.best_params_['C']
    kernel = busqueda.best_params_['kernel']
    degree = busqueda.best_params_['degree']
    print(f"Mejor valor de C: {mejor_C}")
    print(f"Mejor valor de kernel: {kernel}")
    print(f"Mejor valor de kernel: {degree}")
    # mejores_parametros = busqueda.best_params_
    
    # 3. EvaluaciÃ³n del modelo
    # Entrenar el modelo final con el mejor valor de C
    
    modelo = SVC(C=1, kernel='linear', degree=3, probability=True)
    
    #svm_model = SVC(kernel='sigmoid') # Semejante a la regresiÃ³n logÃ­stica y las redes neuronales
    
    
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Evaluar el modelo
    y_prob = modelo.predict_proba(X_test)[:, 1]
    umbral = 0.14
    y_pred2 = (y_prob >= umbral).astype(int)
    
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)

    conf_matrix_df = pd.DataFrame(conf_matrix,
                                   index=['Real Negativo', 'Real Positivo'],  # Orden cambiado
                                   columns=['Predicción Negativa', 'Predicción Positiva'])  # Orden cambiado
    print("\nTabla de Confusión con sklearn:")
    print(conf_matrix_df)
    
    '''
    # 4. ObtenciÃ³n de los pesos del hiperplano (sÃ³lo se puede realizar con SVM lineal)
    pesos = modelo.coef_[0] #Solo si kernel lineal
    intercepto = modelo.intercept_
    print("Intercepto y pesos del hiperplano:")
    print(intercepto, pesos)
    '''
    
def arbolDecision(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    modelo = DecisionTreeClassifier(random_state=42, criterion="log_loss", min_samples_leaf=10, min_samples_split=5) #max_depth = máxima profundidad del árbol
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)

    # Aumentar el umbral de 0.5 a 0.7
    y_prob = modelo.predict_proba(X_test)[:, 1]
    umbral = 0.2
    y_pred2 = (y_prob >= umbral).astype(int)  # Clasificar como 1 si la probabilidad es mayor o igual al umbral

    print(y_pred2)
    conf_matrix = confusion_matrix(y_test, y_pred2)
    print("Tabla de Confusión:")
    print(conf_matrix)
    informe = classification_report(y_test, y_pred2)
    print("Informe de Clasificación:")
    print(informe)

    #Para seleccionar variables si tengo muchas, hacer árbol y las variables que seleccione primero me quedo con ellas, max_depth indicará el número de variables y niveles 
    # Ejemplo de grid de parámetros de búsqueda
    #grid_parametros= {
    #    'criterion': ['gini', 'entropy'],
    #    'max_depth': [None, 5, 10, 15],
    #    'min_samples_split': [2, 5, 10],
    #    'min_samples_leaf': [1, 2, 4]
    #}
    
    # Graficar valores observados frente a valores predichos
    plt.figure(figsize=(10, 6))

    plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Observados')
    plt.scatter(range(len(y_test)), y_pred2, color='orange', label='Valores Predichos')
    plt.xlabel('Índice de la muestra')
    plt.ylabel('Valores')
    plt.title('Valores Observados vs Predichos')

    plt.legend()
    plt.show()
    
def randomForest(X, y):
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear el modelo de Árbol de RandomForest
    modelo = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
    modelo.fit(X_train, Y_train)

    # 3. Evaluación del modelo
    # Obtener pronÃ³sticos
    predicciones = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    umbral = 0.2
    y_pred2 = (y_prob >= umbral).astype(int)

    # Obtener la tabla de confusiÃ³n
    conf_matrix = confusion_matrix(Y_test, y_pred2)
    print("Tabla de Confusión:\n", conf_matrix)

    # Obtener el informe de clasificaciÃ³n
    informe = classification_report(Y_test, y_pred2)
    print("Informe de Clasificación:\n", informe)

    # 4. InterpretaciÃ³n
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

    # 5. Probar distintos parÃ¡metros
    '''modelo = RandomForestClassifier(n_estimators=200, max_features=2, min_samples_split=4, min_samples_leaf=2, random_state=42)
    modelo.fit(X_train, Y_train)
    predicciones = modelo.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, predicciones)
    print("Tabla de Confusión:\n", conf_matrix)
    informe = classification_report(Y_test, predicciones)
    print("Informe de Clasificación:\n", informe)'''

    
    # Graficar valores observados vs predichos
    plt.figure(figsize=(10, 6))

    plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Valores Observados')
    plt.scatter(range(len(Y_test)), predicciones, color='orange', label='Valores Predichos')
    plt.xlabel('Ãndice de la muestra')
    plt.ylabel('Valores')
    plt.title('Valores Observados vs Predichos')

    plt.legend()
    plt.show()


columns = ['age', 'job', 'marital',	'default', 'education', 'balance',	'housing',	'loan',	'contact',	'day',	'month',	'duration',	'campaign',	'pdays',	'previous',	'poutcome',	'y']

# Leer el archivo Excel en un DataFrame
data = pd.read_csv('bank.csv', header=0, names=columns, sep=";")
# Visualizar las primeras filas
print("Primeras filas del conjunto de datos:")
#data = data.sample(frac=0.05, random_state=42)
#Preprocesamiento
data['default'] = data['default'].replace({'no': 0, 'yes': 1})
data['housing'] = data['housing'].replace({'no': 0, 'yes': 1})
data['loan'] = data['loan'].replace({'no': 0, 'yes': 1})

#PROBAR A HACER MEDIA DE BALANCE PARA NULOS
data['balance'] = data['balance'].replace({0: data['balance'].mean()})

data['y'] = data['y'].replace({'no': 0, 'yes': 1})


#4522 registros inicialmente
data = data[data['job'] != 'unknown']
data = data[data['marital'] != 'unknown']
data = data[data['education'] != 'unknown']
#data = data[data['balance'] != 0]


#Variables dummies, no son de utilidad
'''
dummies_drop_first = pd.get_dummies(data['job'], drop_first=True)
data = pd.concat([data, dummies_drop_first], axis=1)
dummies_drop_first = pd.get_dummies(data['marital'], drop_first=True)
data = pd.concat([data, dummies_drop_first], axis=1)
dummies_drop_first = pd.get_dummies(data['education'], drop_first=True)
data = pd.concat([data, dummies_drop_first], axis=1)
dummies_drop_first = pd.get_dummies(data['day'], drop_first=True)
data = pd.concat([data, dummies_drop_first], axis=1)
'''
data = data.drop(columns=['job', 'marital', 'education', 'month', 'contact', 'day', 'default', 'poutcome', 'loan', 'housing'])
#data = data.drop(columns=['job', 'marital', 'education', 'month'])
# Dividir los datos en conjunto de entrenamiento y prueba

X = data[data.columns]
X = X.drop(columns=['y'])
X = eliminarVifAltos(X) 
y = data['y']

regresionLogistica(X, y)

#kNN(X,y)

#SVM(X,y)

#arbolDecision(X,y)

#randomForest(X,y)



