# Practica 2

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Datos
from keras.datasets import boston_housing


#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los conjuntos de entrenamiento y de prueba
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Normalizacion de datos
# valorNormalizado = (valor-media)/desviacionEstandar

# La normalizacion debe realizarse utilizando la media y la desviacion estandar 
# del conjunto de entrenamiento

mean = x_train.mean(axis=0)  # media de cada columna
std = x_train.std(axis=0)    # desviacion estandar de cada columna

x_train -= mean
x_train /= std
x_test -= mean
x_test /= std


############################
## Red neuronal multicapa ## 
############################

# x_train.shape[1] = 13, es el numero de neuronas de entrada 

# Utilizamos la funcion de activacion ReLu para la capa oculta

# No utilizamos ninguna funcion de activacion para la capa de salida porque 
# es un problema de regresion y queremos predecir valores numericos sin transformar

modelo = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(5, activation="relu"),
        Dense(1)
])

# Veamos los detalles del modelo
modelo.summary()

# Compilacion

# El error cuadratico medio (MSE) y el error absoluto medio (MAE) son funciones 
# de perdida comunes utilizadas para problemas de regresion

# MAE no es mas que la media de las diferencias, en valor absoluto, entre
# las predicciones y las observaciones. Por ejemplo, un MAE de 0.5 en este 
# problema significaria que las predicciones se alejan una media de 500 dolares.

# Recordad que las "metrics" solo brindan informacion, mientras que 
# la funcion "loss" juega un papel fundamental en el funcionamiento del algoritmo

modelo.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mae'])

# Entrenamiento del modelo
history = modelo.fit(x_train, y_train, 
                     epochs = 100, 
                     batch_size = 32,
                     validation_split = 0.2)  

# Visualizmos el resultado
print(history.history.keys())
df = pd.DataFrame(history.history)
print(df)
df.plot()

# Representacion grafica
dfMAE = df.loc[:,["mae","val_mae"]]
dfMAE.plot()

dfLoss = df.loc[:,["loss","val_loss"]]
dfLoss.plot()

# Verificamos si el modelo tambien funciona bien en el conjunto de prueba
metrics = modelo.evaluate(x_test, y_test)

# Prediccion
# Obtenemos las predicciones del precio de la vivienda a partir de nuestra red 
# y los comparamos con los valores originales
predicciones = modelo.predict(x_test[0:10,])
print(predicciones)
# Datos reales
y_test[0:10,]

# Dibujamos todas las predicciones
# Tendria que salir una diagonal si la predicion fuera perfecta
predicciones = modelo.predict(x_test)

np.max(predicciones)
np.max(y_test)

fig, ax = plt.subplots()
ax.scatter(predicciones, y_test,  c="blue", label='Modelo')
ax.set_xlabel("Predicciones")
ax.set_ylabel("Observados")
plt.legend(loc="upper left")
plt.plot([0, 60], [0, 60], color='purple')
plt.show()

###########################
#### Guardar el modelo ####
###########################

# Guardamos el modelo
modelo.save('modeloPractica2.keras')

# Cargar el modelo
modelo2 = keras.models.load_model('modeloPractica2.keras')

modelo2.summary()

predicciones2 = modelo2.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(predicciones2, y_test,  c="green", label='Modelo 2')
ax.set_xlabel("Predicciones")
ax.set_ylabel("Observados")
plt.legend(loc="upper left")
plt.plot([0, 60], [0, 60], color='purple')
plt.show()
