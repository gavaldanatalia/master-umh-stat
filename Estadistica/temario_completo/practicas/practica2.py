# Importación de librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, roc_auc_score, roc_curve
import statsmodels.api as sm

# 1. Carga del Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 
           'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean', 
           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 
           'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 
           'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 
           'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst']

# Leer los datos
data = pd.read_csv(url, header=None, names=columns)

# 2. Exploración de los datos
print("Primeras filas del conjunto de datos:")
print(data.head())

print(data.describe())
#ver la cantidad de datos
print(data.shape)

# Información general sobre los datos
print(data.info())
print(data.describe())

# Ver las primeras filas
print(data.head())

#convertir la primera columna "Diagnosis" a variable binaria 
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Dividir los datos en conjunto de entrenamiento y prueba
X = data[['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 
           'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean', 
           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 
           'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 
           'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 
           'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst']]
y = data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#shape del conjunto de entrenamiento y prueba
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Ajustar el modelo de regresión logística
modelo = LogisticRegression(max_iter=1000)  # Aumenta el número de iteraciones
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Obtener los coeficientes del modelo
coefficients = np.append(modelo.intercept_, modelo.coef_)  # Añadir intercepto a los coeficientes
print("\nCoeficientes del modelo:")
print(coefficients)
print(len(coefficients)) #sale 31 porque son 30 variables mas el intercepto (beta0)

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

# Para los datos de test, obtener las probabilidades de predicción con sklearn.
y_prob = modelo.predict_proba(X_test)[:, 1]
print("\nProbabilidades de predicción con sklearn:")
print(y_prob)

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


# Ajustar el modelo usando statsmodels
X_train_const = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Resumen del modelo
print("\nResumen del modelo:")
print(result.summary())
