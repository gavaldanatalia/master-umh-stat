# Practica 5

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from keras.constraints import max_norm

# Tratamiento de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#######################################
## Carga y preprocesamiento de datos ## 
#######################################

path = ''

# Cargamos los datos 
data = pd.read_csv("adult.csv", delimiter=';', header=None, na_values = ' ?')
data.shape

# Establecemos los nombres de columnas 
# (los nombres se pueden encontrar en el archivo adult.names en el repositorio)
data = data.set_axis(['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                      'marital_status', 'occupation', 'relationship', 'race', 
                      'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
                      'native_country', 'income'], axis=1)
data.head(10)

# Identificamos valores perdidos
data.isna().sum()

# Eliminamos las filas con valores perdidos
data.dropna(how = 'any', inplace=True)
data.isna().sum()

data.dtypes

print(data['workclass'].unique())
print(data['education'].unique())
print(data['marital_status'].unique())
print(data['occupation'].unique())
print(data['relationship'].unique())
print(data['race'].unique())
print(data['sex'].unique())
print(data['native_country'].unique())
print(data['income'].unique())

# Separar las variables predictoras de la etiqueta
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X)
print(y)

# Transformamos la etiqueta a numerica
# Como solo hay dos valores, la etiqueta pasa a ser 0 o 1
y = pd.Categorical(y).codes
print(y)

# Repartimos los datos en muestra de entrenamiento (80%) y test (20%), 
# dividiendolos de forma estratificada, utilizando 'y' como etiquetas de clase
# (intentamos mantener la misma proporcion de clases que aparece en el conjunto de datos completo)
# shuffle=True --> mezclamos los datos antes de dividirlos
# Le damos valor a random_statte para obtener una salida reproducible  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y)

# Proporcion de las clases en los diferentes conjuntos
np.unique(y, return_counts=True)[1]/y.shape[0]
np.unique(y_train, return_counts=True)[1]/y_train.shape[0]
np.unique(y_test, return_counts=True)[1]/y_test.shape[0]

# Nombres de variables numericas --> normalizacion
num_var = X.select_dtypes(include=['number']).columns.values
num_var

# Nombres de variables categoricas --> codificacion one-hot
cat_var = X.select_dtypes(include=['object']).columns.values
cat_var

transformer = make_column_transformer(
    (StandardScaler(), num_var),
    (OneHotEncoder(), cat_var),
    verbose_feature_names_out=False)
# Si no ponemos verbose_feature_names_out=False incorpora standardscaler/onehotencoder
# delante del nombre de cada variable

# Ajustamos utilizando informacion del conjunto de entrenamiento
transformer.fit(x_train)

# Datos con variables preprocesadas: pasamos de 14 a 104 columnas
# IMPORTANTE: Al aplicar el transformer a x_train y x_test pasan a ser una matriz dispersa 
# comprimida --> utilizar pd.DataFrame da error; utilizamos pd.DataFrame.sparse.from_spmatrix

# Entrenamiento. Pasamos de 14 a 104 variables!
x_train = transformer.transform(x_train)
x_train = pd.DataFrame.sparse.from_spmatrix(x_train, columns=transformer.get_feature_names_out())  
# Test
x_test = transformer.transform(x_test)
x_test = pd.DataFrame.sparse.from_spmatrix(x_test, columns=transformer.get_feature_names_out())

x_train.columns
x_train.head(5)

x_test.columns
x_test.head(5)

# Para poder pasar los datos a la red neuronal
x_train.dtypes
x_train=x_train.values

x_test.dtypes
x_test=x_test.values

###############################################################
## Red neuronal multicapa con puntos de control en el modelo ## 
###############################################################

modelo = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation='sigmoid')])

# Veamos los detalles del modelo
modelo.summary()

# Planificacion con decrecimiento exponencial de la tasa de aprendizaje 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.1, # tasa de aprendizaje inicial (por defecto en SGD es 0.01)
    decay_steps=500,  
    decay_rate=0.96)

#  Compilacion
modelo.compile(loss = 'binary_crossentropy', 
               optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                              momentum = 0.8),
               metrics = ['accuracy'])

# Nombre con el que se guardara el modelo 
# (si no indicamos ruta se guardara en el directorio actual)
nombreModeloPuntoControl = 'checkpoint.modeloPractica5.keras' 
# Para guardar todos los modelos al final de cada eopca donde haya una mejoria podriamos hacer
#nombreModeloPuntoControl = 'checkpoint.modeloPractica5-{epoch:02d}-{val_accuracy:0.4f}.keras'

# Configuracion: guardamos el modelo al final de cada epoca (opcion por defecto)
# cuando hay una mejora en el accuracy del conjunto de validacion
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                filepath=nombreModeloPuntoControl ,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True,
                                verbose = 1)  # poner a 0 para quitar los mensajes

# Entrenamiento del modelo
# El modelo se guarda al final de cada epoca, si es el mejor hasta el momento
history = modelo.fit(x_train, 
                     y_train, 
                     epochs = 50, 
                     batch_size = 256,
                     validation_split = 0.2,
                     callbacks=[model_checkpoint_callback])  

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
predicciones = modelo.predict(x_test)
print(predicciones[0:10,])
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test = 1*(predicciones>0.5)
print(predic_test[0:10,])

# Datos reales
print(y_test[0:10,])

# Matriz de contingencia o matriz de confusion
mc = confusion_matrix(y_test, predic_test)
print(mc)

# Graficamos la matriz
class_names = ['<=50K', ' >50K']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


############################
## Reguralizacion L1 y L2 ##
############################

# Tenemos las opciones:
# regularizer.L1(l1=0.01)
# regularizer.L2(l2=0.01)
# regularizer.L1L2(l1=0.0, l2=0.0)

modelo2 = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(32, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
        Dense(32, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
        Dense(1, activation='sigmoid')])

# Veamos los detalles del modelo2
modelo2.summary()

# Planificacion con decrecimiento exponencial de la tasa de aprendizaje 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.1, # tasa de aprendizaje inicial (por defecto en SGD es 0.01)
    decay_steps=500,  
    decay_rate=0.96)

#  Compilacion
modelo2.compile(loss = 'binary_crossentropy', 
               optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                              momentum = 0.8),
               metrics = ['accuracy'])

# Entrenamiento del modelo2
history2 = modelo2.fit(x_train, 
                     y_train, 
                     epochs = 50, 
                     batch_size = 256,
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

# Verificamos si el modelo2 tambien funciona bien en el conjunto de prueba
metrics2 = modelo2.evaluate(x_test, y_test)

# Prediccion
predicciones2 = modelo2.predict(x_test)
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test2 = 1*(predicciones2>0.5)

# Matriz de contingencia o matriz de confusion
mc2 = confusion_matrix(y_test, predic_test2)
print(mc2)

# Graficamos la matriz
class_names = ['<=50K', ' >50K']
disp = ConfusionMatrixDisplay(confusion_matrix = mc2, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


####################
## Early stopping ##
####################

modelo3 = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation='sigmoid')])

# Veamos los detalles del modelo3
modelo3.summary()

#  Compilacion
modelo3.compile(loss = 'binary_crossentropy', 
               optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                              momentum = 0.8),
               metrics = ['accuracy'])

# Entrenamiento del modelo3

# "monitor" podria ser loss, accuracy, val_loss o val_accuracy 
# restore_best_weights por defecto es False
# (con False utiliza los pesos obtenidos en el ultimo paso del entrenamiento, 
# no los mejores)
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                           patience = 5,
                                           restore_best_weights=True)

history3 = modelo3.fit(x_train, 
                     y_train, 
                     epochs = 50, 
                     batch_size = 256,
                     validation_split = 0.2,
                     callbacks=[early_stop])  

# Visualizmos el resultado
print(history3.history.keys())
df3 = pd.DataFrame(history3.history)
print(df3)
df3.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy3 = df3.loc[:,["accuracy","val_accuracy"]]
dfAccuracy3.plot()

dfLoss3 = df3.loc[:,["loss","val_loss"]]
dfLoss3.plot()

# Verificamos si el modelo3 tambien funciona bien en el conjunto de prueba
metrics3 = modelo3.evaluate(x_test, y_test)

# Prediccion
predicciones3 = modelo3.predict(x_test)
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test3 = 1*(predicciones3>0.5)

# Matriz de contingencia o matriz de confusion
mc3 = confusion_matrix(y_test, predic_test3)
print(mc3)

# Graficamos la matriz
class_names = ['<=50K', ' >50K']
disp = ConfusionMatrixDisplay(confusion_matrix = mc3, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


################################
## Dropout en capa de entrada ##
################################

# Usamos tambien regularizacion max-norm, lo que garantiza que la norma de 
# los pesos no exceda un valor de, por ejemplo, 3.

modelo4 = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_constraint=max_norm(3)),
        Dense(32, activation="relu", kernel_constraint=max_norm(3)),
        Dense(1, activation='sigmoid')])

# Veamos los detalles del modelo4
modelo4.summary()


#  Compilacion
# Aumentamos el momentum (recomendacion del documento original sobre dropout)
modelo4.compile(loss = 'binary_crossentropy', 
              optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                             momentum = 0.95),    # 0 por defecto 
              metrics = ['accuracy'])

# Entrenamiento del modelo4
history4 = modelo4.fit(x_train, 
                     y_train, 
                     epochs = 50, 
                     batch_size = 256,
                     validation_split = 0.2)

# Visualizmos el resultado
print(history4.history.keys())
df4 = pd.DataFrame(history4.history)
print(df4)
df4.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy4 = df4.loc[:,["accuracy","val_accuracy"]]
dfAccuracy4.plot()

dfLoss4 = df4.loc[:,["loss","val_loss"]]
dfLoss4.plot()

# Verificamos si el modelo4 tambien funciona bien en el conjunto de prueba
metrics4 = modelo4.evaluate(x_test, y_test)

# Prediccion
predicciones4 = modelo4.predict(x_test)
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test4 = 1*(predicciones4>0.5)

# Matriz de contingencia o matriz de confusion
mc4 = confusion_matrix(y_test, predic_test4)
print(mc4)

# Graficamos la matriz
class_names = ['<=50K', ' >50K']
disp = ConfusionMatrixDisplay(confusion_matrix = mc4, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


##############################
## Dropout en capas ocultas ##
##############################

modelo5 = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(32, activation="relu", kernel_constraint=max_norm(3)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_constraint=max_norm(3)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')])

# Veamos los detalles del modelo5
modelo5.summary()

#  Compilacion
# Aumentamos el momentum (recomendacion del documento original sobre dropout)
modelo5.compile(loss = 'binary_crossentropy', 
              optimizer=keras.optimizers.SGD(learning_rate=lr_schedule,
                                             momentum = 0.95),    # 0 por defecto 
              metrics = ['accuracy'])

# Entrenamiento del modelo5
history5 = modelo5.fit(x_train, 
                     y_train, 
                     epochs = 50, 
                     batch_size = 256,
                     validation_split = 0.2)

# Visualizmos el resultado
print(history5.history.keys())
df5 = pd.DataFrame(history5.history)
print(df5)
df5.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy5 = df5.loc[:,["accuracy","val_accuracy"]]
dfAccuracy5.plot()

dfLoss5 = df5.loc[:,["loss","val_loss"]]
dfLoss5.plot()

# Verificamos si el modelo5 tambien funciona bien en el conjunto de prueba
metrics5 = modelo5.evaluate(x_test, y_test)

# Prediccion
predicciones5 = modelo5.predict(x_test)
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test5 = 1*(predicciones5>0.5)

# Matriz de contingencia o matriz de confusion
mc5 = confusion_matrix(y_test, predic_test5)
print(mc5)

# Graficamos la matriz
class_names = ['<=50K', ' >50K']
disp = ConfusionMatrixDisplay(confusion_matrix = mc5, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
