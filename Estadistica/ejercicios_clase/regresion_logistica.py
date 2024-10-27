# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, roc_auc_score, roc_curve
import statsmodels.api as sm

path = '/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/datos/'
df = pd.read_excel(path+'logistica.xlsx')

# Visualizar las primeras filas
print("Primeras filas del conjunto de datos:")
print(df.head())

# Dividir los datos en conjunto de entrenamiento y prueba
X = df[['X1', 'X2']]
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Ajustar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Aumentar el umbral de 0.5 a 0.7
# Estoy siendo más exigente, a partir de 0.7 prediciré 1, más abajo de 0.7 predeciré 0
y_prob = modelo.predict_proba(X_test)[:, 1]
umbral = 0.7
y_pred2 = (y_prob >= umbral).astype(int)  # Clasificar como 1 si la probabilidad es mayor o igual al umbral

# Obtener los coeficientes del modelo
coefficients = np.append(modelo.intercept_, modelo.coef_)  # Añadir intercepto a los coeficientes
print("\nCoeficientes del modelo:")
print(coefficients)


# Tabla de confusión y otras métricas con sklearn
conf_matrix = confusion_matrix(y_test, y_pred)

conf_matrix_df = pd.DataFrame(conf_matrix,
                               index=['Real Negativo', 'Real Positivo'],  # Orden cambiado
                               columns=['Predicción Negativa', 'Predicción Positiva'])  # Orden cambiado
print("\nTabla de Confusión con sklearn:")
print(conf_matrix_df)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))

y_prob = modelo.predict_proba(X_test)[:, 1]
roc_auc=roc_auc_score(y_test, y_prob)
print(f"El área bajo la curva ROC (AUC) es: {roc_auc:.2f}")

# Tabla de confusión de forma manual
true_positive = np.sum((y_test == 1) & (y_pred == 1))
true_negative = np.sum((y_test == 0) & (y_pred == 0))
false_positive = np.sum((y_test == 0) & (y_pred == 1))
false_negative = np.sum((y_test == 1) & (y_pred == 0))

conf_matrix_manual = pd.DataFrame([[true_negative, false_positive],
                                    [false_negative, true_positive]],
                                   index=['Real Negativo', 'Real Positivo'],
                                   columns=['Predicción Negativa', 'Predicción Positiva'])


# Imprimir la tabla de confusión
print("\nTabla de Confusión manual:")
print(conf_matrix_manual)

# Para los datos de test, obtener las probabilidades de predicción manualmente.
X_test_const = sm.add_constant(X_test)  # Agregar la constante al conjunto de prueba
y_prob_manual = 1 / (1 + np.exp(-X_test_const.dot(coefficients)))  # Aplicar la función sigmoide
print("\nProbabilidades de predicción manual:")
print(y_prob_manual)

# Para los datos de test, obtener las probabilidades de predicción con sklearn.
y_prob = modelo.predict_proba(X_test)[:, 1]
print("\nProbabilidades de predicción con sklearn:")
print(y_prob_manual)


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

# Validación de supuestos
# Agregar una constante para el término de intercepto


# Ajustar el modelo usando statsmodels
X_train_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Resumen del modelo
print("\nResumen del modelo:")
print(result.summary())
