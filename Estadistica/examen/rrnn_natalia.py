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


# Lectura de datos
df = pd.read_csv("datos.csv", sep=";")
cols = df.columns

# Identificamos valores perdidos: No hay valores perdidos en ninguna de las variables.
# df.isna().sum()

# Separar las variables predictoras de la etiqueta
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convertimos a númerico la variable que viene en texto ABCD y va a ser la que vamos a predecir
y = pd.Categorical(y).codes
print(y)

# Le damos valor a random_statte para obtener una salida reproducible  
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y)

# Proporcion de las clases en los diferentes conjuntos
# Las clases están bantante balanceadas, todas representan alrededor de un 22% de la muestra en los diferentes conjuntos.
print("Proporción para y", np.unique(y, return_counts=True)[1]/y.shape[0])
print("Proporción para y_train", np.unique(y_train, return_counts=True)[1]/y_train.shape[0])
print("Proporción para y_test", np.unique(y_test, return_counts=True)[1]/y_test.shape[0])


# Nombres de variables numericas --> normalizacion
num_var = X.select_dtypes(include=['number']).columns.values
num_var

# Nombres de variables categoricas --> codificacion one-hot
cat_var = X.select_dtypes(include=['object']).columns.values
cat_var

# Configuramos make_column_transformer para poder transformar las variables
# según el tipo de datos que sea.
transformer = make_column_transformer(
    (StandardScaler(), num_var),
    (OneHotEncoder(), cat_var),
    verbose_feature_names_out=False)
# Si no ponemos verbose_feature_names_out=False incorpora standardscaler/onehotencoder
# delante del nombre de cada variable

# Ajustamos utilizando informacion del conjunto de entrenamiento
transformer.fit(x_train)

# Entrenamiento. Pasamos de 9 a 20 variables
x_train = transformer.transform(x_train)
x_train = pd.DataFrame(x_train)

# Test. Hacemos la transformación de las variables, igual que con el train
x_test = transformer.transform(x_test)
x_test = pd.DataFrame(x_test)

# Comproamos los nombres de las columnas y vemos los df para que todo esté bien.
x_train.columns
x_train.head(5)

x_test.columns
x_test.head(5)

# Convertimos a one hot para 4 clases que son las que tenemos que predecir
y_test = keras.utils.to_categorical(y_test, num_classes=4)
y_train = keras.utils.to_categorical(y_train, num_classes=4)


# Tenemos 4 capas, la primera de ellas es la inicial con 20 (número de columnas)
# neuronas. El número de neuronas ocultas son 32 y 32 que forman las capas ocultas.
# Luego cogemos el tipo de función softmax de activación puesto que estmaos en un 
# problema de predicción de clasificación múltiple. Indicamos también el número de clases
# a predecir en esta última capa
modelo = Sequential([
        Input(shape=(x_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(4, activation='softmax')])

# Veamos los detalles del modelo
modelo.summary()

# Planificacion con decrecimiento exponencial de la tasa de aprendizaje 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.1, # tasa de aprendizaje inicial
    decay_steps=500,  
    decay_rate=0.96)

# Compilacion
# Cogemos el optimizador adam, podriamos haber cogido otro como SDG
# La métrica por la que nos vamos a regir va a ser el accuracy
# La función de perdida es categorical_crossentropy
modelo.compile(loss = 'categorical_crossentropy', 
               optimizer='adam',
               metrics = ['accuracy'])

# Early stoping con una "paciencia" de 5
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                           patience = 5,
                                           restore_best_weights=True)

# Entrenamiento del modelo con 100 epocas, puesto que con 30, 20, 100 no acababa de aprender bien
# Ponemos early stoping para ver si combatimos el sobreaprendizaje
history = modelo.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=[early_stop])

modelo.save('GavaldaNataliainicial.keras')

# Evaluación del modelo
loss, accuracy = modelo.evaluate(x_test, y_test)
print(f"Precisión en test: {accuracy:.4f}")

# Visualizmos el resultado y lo guardamos en la variable history
# Esto nos permite hacer los graficos siguientes
print(history.history.keys())
history = pd.DataFrame(history.history)
print(history)
history.plot()

# Representamos por separado la evolucion de la funcion de perdida y el accuracy
dfAccuracy = history.loc[:,["accuracy","val_accuracy"]]
dfAccuracy.plot()

dfLoss = history.loc[:,["loss","val_loss"]]
dfLoss.plot()

# Lo hacemos para todo el conjunto de test
predicciones = modelo.predict(x_test)

# Obtenemos la categoria donde se produce el maximo de la probabilidad de clasificacion
predic_test = np.argmax(predicciones, axis=1)
originales_test = np.argmax(y_test, axis=1) 

# Matriz de contingencia o matriz de confusion
conf_matrix = confusion_matrix(originales_test, predic_test)
    
# Print de la matriz de confusion
class_names = ['0','1','2','3']
disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

def accuracy_by_classs(conf_matrix):
    """
    Calcula el accuracy por clase a partir de una matriz de confusión.
    """
    class_acc = {}
    
    for i in range(len(conf_matrix)):  # Para cada clase
        TP = conf_matrix[i, i]  # Verdaderos positivos (diagonal)
        total_real = np.sum(conf_matrix[i, :])  # Total de ejemplos de esa clase (suma de la fila)
        
        accuracy = TP / total_real if total_real != 0 else 0  # Evitar división por 0
        class_acc[f"Clase {i}"] = accuracy
    
    return class_acc

# Accuracy por clase
accuracy_per_class = accuracy_by_classs(conf_matrix)

# Mostrar resultados
for clase, acc in accuracy_per_class.items():
    print(f"{clase}: {acc:.2%}")
    
    
#### Conclusiones
# Con el modelo simple ha tendido a sobreaprender. Lo he detectado porque el accuracy del cojunto de test vs el conjunto de train ha sido muy dispar. Además también se puede ver en la gráfica, donde la línea naranja se va separando más todavía de la línea azul. Si no hubiera sobreaprendizaje, ambas líneas tendrían que estar casi solapadas. 
# Eso querría decir que no hay sobreaprendizaje y que el moddelo es capaz de generalizar bien.
# Lo voy a abordar poniendo una tasa de aprendizaje al 0.1 y añadiendo early stopping.
# Accuracy de mi modelo: 0.556408 y 0.56 de train y test
# La matriz de confusión se muestra anteriormente y el accuracy por clase es el siguiente:
# - Clase 0: 40.18%
# - Clase 1: 25.55%
# - Clase 2: 61.96%
# - Clase 3: 73.73%
# La que peor predice es la clase 0 y 1. 
# Parece que el modelo puede estar confundiendo también las clases 1 y 2 debido a que se ven que muchos casos que realmente es un tipo de cliente 1 se ha etiquetado como 2 y viceversa. Eso también pasa con la clase 0 y la clase 3. 