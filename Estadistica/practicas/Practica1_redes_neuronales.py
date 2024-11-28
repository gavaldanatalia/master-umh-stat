# Practica 1

# conda install tensorflow
# onda install sklearn


# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Datos
from keras.datasets import mnist

# Matriz de confusion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los conjuntos de entrenamiento y de prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Echemos un vistazo a los datos de entrenamiento:
# Tenemos 60000 imagenes de 28 x 28. Los datos son enteros de 8 bits
print(x_train.ndim)
print(x_train.shape)
print(x_train.dtype)

print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Veamos el reparto de cada uno de los digitos en los datos de entrenamiento:
np.unique(y_train, return_counts=True)

# Visualizamos la primera imagen del conjunto de entrenamiento
plt.imshow(x_train[0], cmap=plt.cm.binary)
print("Etiqueta asignada: ", y_train[0])

# Podemos ver tambien la matriz de intensidades de cada uno de los 28x28 pixeles 
# con valores entre [0, 255] que representa esta primera muestra 
print(np.matrix(x_train[0]))

# Nuestros datos son de la forma (imagenes, ancho, alto).
# A la red solo puede entrar una dimension --> necesitamos "aplanar" los datos
x_train = x_train.reshape(x_train.shape[0], 784)   # (6000, 28*28)
x_test = x_test.reshape(x_test.shape[0], 784)      # (1000, 28*28)

# Los valores de los pixeles pueden tomar valores de 0 (blanco) a 255 (negro)
np.min(x_train)
np.max(x_train)

np.min(x_test)
np.max(x_test)

# Pasamos a escala [0,1]
x_train = x_train/255
x_test = x_test/255

np.max(x_train)
np.max(x_test)

# Veamos las etiquetas
print(y_train[1:10])
print(y_test[1:10])

# Codificacion one-hot con la funcion to_categorical()
# El vector de etiquetas a convertir debe ser un vector de enteros COMENZANDO EN 0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(y_train[1:10,])
print(y_test[1:10,])

############################
## Red neuronal multicapa ## 
############################

# 28*28 = 784 neuronas de entrada
# tenemos tantas entradas como pixeles tiene una imagen (28*28)
# softmax --> clasificacion multiclase

# Definimos un modelo secuencial
modelo = Sequential()

# Agregamos las capas

# La decimos a la red que forma de entrada debe esperar (no es obligatorio)
modelo.add(Input(shape=(784,)))

# Agregamos una capa densamente conectada al modelo
modelo.add(Dense(256, activation='relu'))

# En versiones anteriores de keras se solia incluir un argumento input_shape 
# en la primera capa. Puede hacerse pero aparecera un warning
# modelo.add(Dense(256, activation='relu', input_shape=(784,)))

# La capa de salida tiene 10 neuronas (10 clases)           
modelo.add(Dense(10, activation='softmax'))

# Veamos los detalles del modelo
modelo.summary()

######## Por que tenemos un total de 203530 parametros =  200960 + 2570?

# Compilacion

# La funcion de perdida es categorical_crossentropy
# Aplicamos el optimizador rmsprop para encontrar los pesos y sesgos 
# Tambien indicamos las metricas que el modelo evaluara durante el entrenamiento

# Las "metrics" solo brindan informacion, mientras que la funcion "loss" 
# juega un papel fundamental en el funcionamiento del algoritmo

modelo.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Entrenamiento del modelo
# En cada epoca durante el entrenamiento, Keras muestra el numero de instancias 
# procesadas hasta el momento (junto con una barra de progreso), 
# el tiempo medio de entrenamiento, la perdida (loss) y la precision (accuracy), 
# ambas calculadas en el conjunto de entrenamiento y el conjunto de validacion.

# Si guardamos el entrenamiento (por ejemplo, en la variable history)
# podremos visualizar las metricas del modelo
history = modelo.fit(x_train, y_train, 
                     epochs = 10, 
                     batch_size = 128,
                     validation_split = 0.2)  

########## Por que las epocas llegan hasta 375? 


# Visualizmos el resultado
print(history.history.keys())
df = pd.DataFrame(history.history)
print(df)
df.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy = df.loc[:,["accuracy","val_accuracy"]]
dfAccuracy.plot()

dfLoss = df.loc[:,["loss","val_loss"]]
dfLoss.plot()

# Verificamos si el modelo tambien funciona bien en el conjunto de prueba
metrics = modelo.evaluate(x_test, y_test)

# Prediccion
predicciones = modelo.predict(x_test[1:10,])
print(predicciones)
predicciones[0].shape
np.sum(predicciones[0])
np.argmax(predicciones[0])
# La categoria a la que se asigna la primera muestra de test es la que se encuentra 
# en la posicion 2 con probabilidad
predicciones[0][2]

# Obtenemos la categoria donde se produce el maximo de la probabilidad de clasificacion
predic_test = np.argmax(predicciones, axis=1)
print(predic_test)

# Datos reales
y_test[1:10,]
np.argmax(y_test[1:10,], axis=1)

# Lo hacemos para todo el conjunto de test
predicciones = modelo.predict(x_test)
# Obtenemos la categoria donde se produce el maximo de la probabilidad de clasificacion
predic_test = np.argmax(predicciones, axis=1)
originales_test = np.argmax(y_test, axis=1)

# Matriz de contingencia o matriz de confusion
mc = confusion_matrix(originales_test, predic_test)
print(mc)

# Graficamos la matriz
class_names = ['0','1','2','3','4','5','6','7','8','9']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

#################
## Ejercicio 1 ##  
#################

# Probamos con un numero escrito a mano por nosotros 

from skimage import io
num = io.imread('siete.png',as_gray=True) 

num = abs(1-num)  
plt.imshow(num, cmap=plt.cm.binary)

print(num.shape)  # debe ser (28, 28)

resultado = modelo.predict(num.reshape(1, 784))
resultado
resultado = np.argmax(resultado, axis=1)
print(resultado)

