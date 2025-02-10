# Practica 3

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input

# Tratamiento de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los datos (indicar header=None si los datos no tuvieran cabecera)
data = pd.read_csv("stroke.csv", delimiter=',')
data.shape
data.head(10)

# Eliminamos la columna ID
data = data.drop(columns = 'id')

# Identificamos valores perdidos
data.isna().sum()

# Eliminamos las filas con valores perdidos
data.dropna(how = 'any', inplace=True)
data.isna().sum()

data.dtypes

print(data['gender'].unique())
print(data['hypertension'].unique())
print(data['heart_disease'].unique())
print(data['ever_married'].unique())
print(data['work_type'].unique())
print(data['Residence_type'].unique())
print(data['smoking_status'].unique())
print(data['stroke'].unique())

# Supongamos, por ejemplo, que gender tomara los valores 0, 1 y 2 (Male, Female, Other)
# Una posibilidad para pasarla a tipo “object” es: 
# data['gender'] = data.gender.astype(str)

# Separamos las variables predictoras de la etiqueta
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X)
print(y)

# Transformamos la etiqueta a numerica
# Como solo hay dos valores, la etiqueta pasa a ser 0 o 1
# Si fueran 3 valores, pasarian a ser 0,1,2 ...
y = pd.Categorical(y).codes
print(y)

# Importante: si optamos por una representacion 0/1 en el problema de 
# clasificacion binaria, utilizaremos una unica neurona sigmoidal en la capa de salida
# Esta es la opcion escogida aqui

# Tambien podriamos haber hecho una codificacion one-hot 
# y = keras.utils.to_categorical(y, num_classes=2)
# con lo que tendriamos que utilizar una capa de salida softmax con dos neuronas de salida

# Ambas alternativas funcionarian igual de bien, aunque lo mas habitual es 
# quedarnos con una sola neurona, por ser la alternativa mas simple


# Repartimos los datos en muestras de entrenamiento (80%) y test (20%), 
# dividiendolos de forma estratificada, utilizando 'y' como etiquetas de clase
# (intentamos mantener la misma proporcion de clases que aparece en el conjunto de datos completo)
# shuffle=True --> mezclamos los datos antes de dividirlos
# Le damos valor a random_statte para obtener una salida reproducible  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y)


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

# Datos con variables preprocesadas: pasamos de 10 a 23 columnas
# transformer.get_feature_names_out() tiene los nombres de las columnas
# Entrenamiento
x_train = transformer.transform(x_train)
x_train = pd.DataFrame(x_train, columns=transformer.get_feature_names_out())
x_train.shape

# Test
x_test = transformer.transform(x_test)
x_test = pd.DataFrame(x_test, columns=transformer.get_feature_names_out())

x_train.columns
x_train.head(5)

x_test.columns
x_test.head(5)

############################
## Red neuronal multicapa ## 
############################

def create_model(neuronasEntrada):
    
    # Utilizamos la funcion de activacion ReLu en la capa oculta
    # Tenemos un problema de clasificacion binaria 0/1 -> neurona de salida sigmoid
    
    modelo = Sequential([
            Input(shape=(neuronasEntrada,)),
            Dense(8, activation="relu"),
            Dense(1, activation='sigmoid')
    ])
    
    #  Compilacion
    
    # La funcion de perdida binary_crossentropy se usa para clasificacion binaria 
    # Recordad que usamos categorical_crossentropy para clasificacion en mas de dos grupos
    # Aplicamos el optimizador sgd para encontrar los pesos y sesgos 
    # Tambien indicamos las metricas que el modelo evaluara durante el entrenamiento
    
    modelo.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    return modelo

# x_train.shape[1] es el numero de neuronas de entrada 
modelo = create_model(x_train.shape[1])
    
# Veamos los detalles del modelo
modelo.summary()

# Entrenamiento del modelo
history = modelo.fit(x_train, 
                     y_train, 
                     epochs = 20, 
                     batch_size = 32)  

# Visualizmos el resultado
print(history.history.keys())
df = pd.DataFrame(history.history)
print(df)
df.plot()

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
class_names = ['NoStroke', 'Stroke']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

# Aunque el modelo proporciona buenos resultados globales, 
# no lo hace bien para el grupo de enfermos.

np.unique(y, return_counts=True)
np.unique(y, return_counts=True)[1]/y.shape[0]

np.unique(y_train, return_counts=True)[1]/y_train.shape[0]
np.unique(y_test, return_counts=True)[1]/y_test.shape[0]

# Una posibilidad es ampliar el banco de datos muestreando sobre los enfermos 
# para ampliar la muestra

# Dimension original de los datos
data.shape
# Remuestreamos con reemplazamiento sobre los enfermos
filtro = data['stroke'] == 'Yes'
# Valores originales
print("Originales: ",sum(filtro),"\n")
nuevos = data[filtro].sample(n=2000, replace=True)
# Datos ampliados
data_ampliado = pd.concat([data, nuevos])
data_ampliado.shape

# Generamos de nuevo las divisones de entrenamiento y prueba
# y hacemos el tratamiento de variables numericas y categoricas

X = data_ampliado.iloc[:, :-1]
y = data_ampliado.iloc[:, -1]

# Transformamos la etiqueta en 0 y 1
y = pd.Categorical(y).codes

  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y)

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

# Ajustamos utilizando informacion del conjunto de entrenamiento
transformer.fit(x_train)

# Datos con variables preprocesadas: pasamos de 10 a 23 columnas
# Entrenamiento
x_train = transformer.transform(x_train)
x_train = pd.DataFrame(x_train, columns=transformer.get_feature_names_out())
# Test
x_test = transformer.transform(x_test)
x_test = pd.DataFrame(x_test, columns=transformer.get_feature_names_out())


# Analizamos el comportamiento de la red neuronal con los datos ampliados
modelo2 = create_model(x_train.shape[1])

# Entrenamiento del modelo
history2 = modelo2.fit(x_train, y_train, 
                     epochs = 20, 
                     batch_size = 32)  

# Visualizmos el resultado
print(history2.history.keys())
df2 = pd.DataFrame(history2.history)
print(df2)
df2.plot()

# Verificamos si el modelo2 tambien funciona bien en el conjunto de prueba
metrics2 = modelo2.evaluate(x_test, y_test)

# Prediccion
predicciones2 = modelo2.predict(x_test)
# Obtenemos la etiqueta donde se produce el maximo de la probabilidad de clasificacion
predic_test2 = 1*(predicciones2>0.5)

# Matriz de contingencia o matriz de confusion
mc = confusion_matrix(y_test, predic_test2)
print(mc)

# Graficamos la matriz
class_names = ['NoStroke', 'Stroke']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

# Continuaremos trabajando con el conjunto de datos ampliado

####################################################
#### Division automatica de datos de validacion ####
####################################################

# Particion automatica de 80% para train y 20% para validacion 
# Keras divide los datos por orden, tomando el primer 80% de los datos como 
# datos de entrenamiento y el ultimo 20% como datos de validacion
# Cuidado: el uso de validation_split no conserva el porcentaje de cada clase

modelo3 = create_model(x_train.shape[1])

# Entrenamiento del modelo
history3 = modelo3.fit(x_train, y_train, 
                     epochs = 20, 
                     batch_size = 32,
                     validation_split = 0.2)  

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
mc = confusion_matrix(y_test, predic_test3)
print(mc)

# Graficamos la matriz
class_names = ['NoStroke', 'Stroke']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')


################################################
#### Division manual de datos de validacion ####
################################################

# Particion manual del 80% para train y 20% para validacion
x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y_train)

modelo4 = create_model(x_train2.shape[1])

# Entrenamiento del modelo
history4 = modelo4.fit(x_train2, y_train2, 
                     epochs = 20, 
                     batch_size = 32,
                     validation_data=(x_val, y_val))  

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
mc = confusion_matrix(y_test, predic_test4)
print(mc)

# Graficamos la matriz
class_names = ['NoStroke', 'Stroke']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

############################
#### Valizacion cruzada ####
############################

# Creamos y evaluamos 5 modelos utilizando 5 divisiones del conjunto de entrenamiento

# Podemos desactivar la salida detallada de la funcion fit() utilizando verbose = 0  

# El rendimiento promedio de los 5 modelos se calcula al final y nos proporciona 
# una estimacion del accuracy del modelo

k = 5
num_validation_samples=len(x_train)/k
accuracy_scores=[]
loss_scores=[]

for fold in range(k):
    
    print("Iteracion: ", fold+1)
    
    # Muestras de entrenamiento y validacion para la particion k
    ini = int(num_validation_samples*fold)
    fin = int(num_validation_samples*(fold+1))
        
    # Conjunto de validacion 
    x_val = x_train[ini:fin]
    y_val = y_train[ini:fin]
    
    # Conjunto de entrenamiento (el resto de datos de nuestro 'train' inicial)
    x_train2 = np.concatenate([x_train[:ini], x_train[fin:]])
    y_train2 = np.concatenate([y_train[:ini], y_train[fin:]])
   
    # Creamos el modelo
    modeloCV = create_model(x_train2.shape[1]) 
    
    # Entrenamiento del modelo
    historyCV = modeloCV.fit(x_train2, y_train2, 
                         epochs = 20, 
                         batch_size = 32,
                         verbose = 0)  
    
    # Guardamos los resultados en el conjunto de validacion    
    val_loss, val_acc = modeloCV.evaluate(x_val, y_val, verbose = 0)
    accuracy_scores.append(val_acc)
    loss_scores.append(val_loss)


final_accuracy = np.mean(accuracy_scores)
sd_accuracy = np.std(accuracy_scores)
final_loss = np.mean(loss_scores)
sd_loss = np.std(loss_scores)

print("Loss validacion cruzada: ", np.round(final_loss,2) , " Desviacion tipica: ", np.round(sd_loss,4))
print("Accuracy validacion cruzada: ", np.round(final_accuracy,2) , " Desviacion tipica: ", np.round(sd_accuracy,4))

# Parece que la solucion alcanzada con el modelo es bastante estable

# Finalmente, comparamos los resultados obtenidos en el conjunto de prueba
print(metrics4)

#############################################
#### Valizacion cruzada con scikit-learn ####   
#############################################

from sklearn.model_selection import StratifiedKFold

k = 5
accuracy_scores2=[]
loss_scores2=[]

# barajamos los datos y establecemos una semilla 
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state = 123)

print(x_train)
print(y_train)


for train,val in kfold.split(x_train,y_train):
        
    # Creamos el modelo
    modeloCV2 = create_model(x_train.shape[1]) 
    
    # Entrenamiento del modelo
    historyCV2 = modeloCV2.fit(x_train.iloc[train,:], y_train[train], 
                         epochs = 20, 
                         batch_size = 32,
                         verbose = 0)  
    
    # Guardamos los resultados en el conjunto de validacion
    val_loss, val_acc = modeloCV2.evaluate(x_train.iloc[val,:], y_train[val], verbose = 0)
    accuracy_scores2.append(val_acc)
    loss_scores2.append(val_loss)

final_accuracy2 = np.mean(accuracy_scores2)
sd_accuracy2 = np.std( accuracy_scores2)
final_loss2 = np.mean(loss_scores2)
sd_loss2 = np.std(loss_scores2)

print("Loss validacion cruzada: ", np.round(final_loss2,2) , " Desviacion tipica: ", np.round(sd_loss2,4))
print("Accuracy validacion cruzada: ", np.round(final_accuracy2,2) , " Desviacion tipica: ", np.round(sd_accuracy2,4))
