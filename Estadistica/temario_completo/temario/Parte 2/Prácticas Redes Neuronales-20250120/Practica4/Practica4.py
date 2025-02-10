# Practica 4

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Tratamiento de datos 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

# Datos
from keras.datasets import reuters

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los conjuntos de entrenamiento y de prueba
# Conservamos las 10.000 palabras mas frecuentes
(x_train, y_train), (x_test, y_test)  = reuters.load_data(num_words=10000)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Cada noticia es una lista de numeros enteros (indices de palabras)
x_train[0]

# La etiqueta asociada con una noticia es un numero entero entre 0 y 45 (indice de tema)
y_train[0]

# Decodificamos el conjunto de datos en las cadenas de texto correspondientes (en ingles)
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Mostramos el primer elemento
# Los indices estan desplazados 3 posiciones porque 0, 1 y 2 son indices 
# reservados para “relleno”, “inicio de secuencia” y “desconocido”
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in x_train[0]])
decoded_newswire

# Funcion que nos permite vectorizar cada secuencia de texto en un vector de 0's y 1's
def vectorize_sequences(sequences, dimension=10000):
  # Creates an all-zero matrix of shape (len(sequences),dimension)
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
       for j in sequence:
           results[i, j] = 1. # Sets specific indices of results[i] to 1s
  return results

# Vectorized training and test data
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# Ahora este es el aspecto de las muestras de entrenamiento y test
x_train[0]

# Codificacion one-hot con la funcion to_categorical()
np.min(y_train)
np.max(y_train)

# Si no indicamos "num_classes", sera max(datos)+1
y_train = keras.utils.to_categorical(y_train, num_classes=46)
y_test = keras.utils.to_categorical(y_test, num_classes=46)

print(y_train[0:5,])
print(y_test[0:5,])


############################
## Red neuronal multicapa ## 
############################

def create_model(neuronasEntrada):
    
    modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(46, activation='softmax')
    ])
    
    return modelo

# x_train.shape[1] es el numero de neuronas de entrada
x_train.shape[1] 
modelo = create_model(x_train.shape[1])
    
# Veamos los detalles del modelo
modelo.summary()

#  Compilacion    
modelo.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Entrenamiento del modelo
history = modelo.fit(x_train, y_train, 
                     epochs = 100, 
                     batch_size = 512,
                     validation_split = 0.2)  

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

# Verificamos si el modelo funciona bien en el conjunto de prueba
metrics = modelo.evaluate(x_test, y_test)

# Prediccion
predicciones = modelo.predict(x_test)
# Obtenemos la categoria donde se produce el maximo de la probabilidad de clasificacion
predic_test = np.argmax(predicciones, axis=1)

originales_test = np.argmax(y_test, axis=1)
pd.DataFrame(originales_test).value_counts()

# Matriz de contingencia o matriz de confusion 
mc = confusion_matrix(originales_test, predic_test)
print(mc)

# Graficamos la matriz. No se ve nada al ser muchas clases
disp = ConfusionMatrixDisplay(confusion_matrix = mc)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


###############################################################
## Red neuronal multicapa con tasa de aprendizaje adaptativa ## 
###############################################################

# La aquitectura no cambia
modelo2 = create_model(x_train.shape[1])
    
# Veamos los detalles del modelo
modelo2.summary()

# Compilacion

# Creamos una planificacion con decrecimiento exponencial de la tasa de aprendizaje 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.1, # tasa de aprendizaje inicial (por defecto en SGD es 0.01)
    decay_steps=100,  
    decay_rate=0.96)

momento = 0.8

modelo2.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum = momento),
              metrics=['accuracy'])


# Entrenamiento del modelo
history2 = modelo2.fit(x_train, y_train, 
                     epochs = 100, 
                     batch_size = 512,
                     validation_split = 0.2)  

# Visualizmos el resultado
print(history2.history.keys())
df2 = pd.DataFrame(history2.history)
print(df2)
df2.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy2 = df2.loc[:,["accuracy","val_accuracy"]]
dfAccuracy2.plot()

dfLoss2 = df2.loc[:,["loss","val_loss"]]
dfLoss2.plot()

# Verificamos si el modelo3 tambien funciona bien en el conjunto de prueba
metrics2 = modelo2.evaluate(x_test, y_test)

# Comparamos los resultados con el modelo anterior
metrics


############################
### Busqueda sistematica ###
############################

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Funcion para crear el modelo, necesaria para KerasClassifier
def create_model(neuronasEntrada, neurons): 
    
    modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(neurons, activation="relu"),
            Dense(neurons, activation="relu"),
            Dense(46, activation='softmax')
    ])
       
    return modelo

# El constructor de la clase KerasClassifier puede tomar argumentos predeterminados 
# que se pasan a las llamadas de compile () y fit(), como la funcion de perdida,
# las metricas, la cantidad de epocas o el batch_size
modelGridSearch = KerasClassifier(model=create_model, 
                                  loss='categorical_crossentropy', metrics=['accuracy'])

# Definimos los parametros de la busqueda 
# 3x3x3x3 = 81 combinaciones de paramteros
neuronas = [16, 32, 64]
opt = ['sgd', 'rmsprop', 'adam']
epocas= [25,50,100]
lotes = [128, 256, 512]

# El prefijo model__ en el diccionario de parametros param_grid es necesario para que 
# KerasClassifier sepa que el parametro debe pasarse a la funcion create_model() como argumento,
# en lugar de algun parametro para configurar en compile() o fit()
param_grid = dict(model__neuronasEntrada = [x_train.shape[1]], # utilizamos [] porque espera una lista
                  model__neurons=neuronas,
                  optimizer=opt,
                  epochs=epocas, 
                  batch_size=lotes,
                  verbose = [0]) # utilizamos [] porque espera una lista

# cv debe ser un entero en [2, inf). Por defecto es 5 (5-fold cross validation)
# 3x3x3x3 = 81 combinaciones de parametros x 2 folds = 162
grid = GridSearchCV(estimator=modelGridSearch, 
                    param_grid=param_grid, cv = 2, verbose=3)
grid_result = grid.fit(x_train, y_train)

# Resumen resultados
print("Mejor: %f utilizando %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) con: %r" % (mean, stdev, param))

# Guardamos el mejor modelo encontrado
bestmodel = grid_result.best_estimator_.model_
bestmodel.save('modeloPractica4GridSearchCV.keras')
metricsGridSearchCV = bestmodel.evaluate(x_test, y_test)    
    
# RandomizedSearchCV
# A diferencia de GridSearchCV, no se prueban todos los valores de los parametros, 
# sino que se toma una muestra de un numero fijo de configuraciones de parametros. 
# La cantidad de configuraciones de parametros que se prueban se indica mediante n_iter.    
    
from sklearn.model_selection import RandomizedSearchCV

# 10 combinaciones de parametros x 2 folds = 20
random_search = RandomizedSearchCV(estimator=modelGridSearch, 
                                   param_distributions=param_grid, cv = 2, verbose=3,
                                   n_iter = 10)
random_search_result = random_search.fit(x_train, y_train)

# Resumen resultados
print("Mejor: %f utilizando %s" % (random_search_result.best_score_, random_search_result.best_params_))
means = random_search_result.cv_results_['mean_test_score']
stds = random_search_result.cv_results_['std_test_score']
params = random_search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) con: %r" % (mean, stdev, param))
    
# Guardamos el mejor modelo encontrado
bestmodelRandom = random_search_result.best_estimator_.model_
bestmodelRandom.save('modeloPractica4RandomizedSearchCV.keras')
metricsRandomizedSearchCV = bestmodelRandom.evaluate(x_test, y_test) 

