# Practica 7

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Reshape, LSTM

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los datos (indicar header=None si los datos no tuvieran cabecera)
data = pd.read_csv("jena_climate_2009_2016.csv", delimiter=',')
data.shape

# Cada linea es un instante temporal: un registro de fecha y 14 valores relacionados con ese instante
data.head(10)

# Identificamos valores perdidos
data.isna().sum()

data.dtypes

# Descartamos la columna Data Time
data = data.drop(columns = 'Date Time')

# Guardamos la temperatura (en grados centigrados)
temperature = data.iloc[:, 1]

# Grafico de la temperatura a lo largo del tiempo. Los datos abarcan 8 años
plt.plot(range(len(temperature)), temperature)

# Grafico mas reducido de los datos de temperatura de los 10 primeros dias 
# Como los datos se registran cada 10 minutos, se obtienen 24×6 = 144 puntos de datos por dia
plt.plot(range(1440), temperature[:1440]);


# 50% entrenamiento, 25% validacion, 25% test
num_train_samples = int(0.5 * len(data))
num_val_samples = int(0.25 * len(data))
num_test_samples = len(data) - num_train_samples - num_val_samples

# Todos nuestros datos son numericos --> normalizacion
# Calculamos la media y la desviacion estandar utilizando solo los datos de entrenamiento
mean = data[:num_train_samples].mean(axis=0)
std = data[:num_train_samples].std(axis=0)
# Normalizacion
data -= mean
data /= std

# Utilizamos timeseries_dataset_from_array() para crear tres conjuntos de datos: 
# uno para el entrenamiento, otro para la validacion y otro para test

# Solo conservamos un punto de datos de cada 6 (un punto por hora)
sampling_rate = 6 
# Las observaciones se remontan a 5 dias (120 horas)
sequence_length = 120 
# El objetivo de una secuencia sera la temperatura 24 horas despues del final de la secuencia
delay = sampling_rate * (sequence_length + 24 - 1) 
# Trabajaremos con lotes de 256 muestras
batch_size = 256


train_dataset = keras.utils.timeseries_dataset_from_array(
    data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples) # utilizamos el primer 50% de los datos para entrenamiento

val_dataset = keras.utils.timeseries_dataset_from_array(
    data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples) # 25% siguiente para validacion

test_dataset = keras.utils.timeseries_dataset_from_array(
    data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples) # ultimo 25% para test

# Veamos la salida de uno de nuestros conjuntos de datos
for muestras, objetivos in train_dataset:
    print("muestras shape:", muestras.shape)
    print("objetivos shape:", objetivos.shape)
    break

muestras[0]  # 120 horas consecutivas de datos de entrada
objetivos[0] # temperatura objetivo para esa muestra

# Las muestras se barajan aleatoriamente, por lo que dos secuencias consecutivas de un lote 
# (como muestras[0] y muestras[1]) no estan necesariamente proximas en el tiempo
muestras[1]  # 120 horas consecutivas de datos de entrada
objetivos[1] # temperatura objetivo para esa muestra
    

#######################################
## Red neuronal densamente conectada ## 
#######################################

# Con Reshape aplanamos los datos a una forma de sequence_length * num_features (120 * 14 = 1680)

modelo = Sequential([
        Input(shape=(sequence_length, data.shape[1])), # (120, 14)
        Reshape((sequence_length * data.shape[1],)),   # Aplanar 
        Dense(16, activation="relu"),
        Dense(1)
    ])

modelo.summary()


#  Compilacion
modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


modelo_checkpoint_callback = keras.callbacks.ModelCheckpoint("jena_dense.keras",
                                                            save_best_only=True)

# Entrenamiento del modelo
history = modelo.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=[modelo_checkpoint_callback])

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
metrics = modelo.evaluate(test_dataset)


#############################
## Red neuronal recurrente ## 
#############################

modelo1 = keras.Sequential([
    Input(shape=(sequence_length, data.shape[1])),
    LSTM(16),
    Dense(1)
])

modelo1.summary()

#  Compilacion
modelo1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


modelo1_checkpoint_callback = keras.callbacks.ModelCheckpoint("jena_lstm.keras",
                                                            save_best_only=True)

# Entrenamiento del modelo
history1 = modelo1.fit(train_dataset,
                    epochs=10,   
                    validation_data=val_dataset,
                    callbacks=[modelo1_checkpoint_callback])

# Visualizmos el resultado
print(history1.history.keys())
df1 = pd.DataFrame(history1.history)
print(df1)
df1.plot()

# Representacion grafica
dfMAE1 = df1.loc[:,["mae","val_mae"]]
dfMAE1.plot()

dfLoss1 = df1.loc[:,["loss","val_loss"]]
dfLoss1.plot()


# Verificamos si el modelo tambien funciona bien en el conjunto de prueba
metrics = modelo1.evaluate(test_dataset)




