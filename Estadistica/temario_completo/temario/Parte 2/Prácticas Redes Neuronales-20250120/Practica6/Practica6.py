# Practica 6

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Módulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten

# Datos
from keras.datasets import fashion_mnist

# Matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los conjuntos de entrenamiento y de prueba
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Echemos un vistazo a los datos de entrenamiento:
# Tenemos 60000 imagenes de 28 x 28. Los datos son enteros de 8 bits
print(x_train.ndim)
print(x_train.shape)
print(x_train.dtype)

print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# Veamos el reparto de cada una de las prendas en los datos de entrenamiento:
np.unique(y_train, return_counts=True)

# Visualizamos la primera imagen del conjunto de entrenamiento
plt.imshow(x_train[0], cmap=plt.cm.binary)
print("Etiqueta asignada: ", y_train[0])

# Las redes convolucionales esperan como entrada las dimensiones [muestras, ancho, alto, canales]
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# Los valores de los pixeles pueden tomar valores de 0 (blanco) a 255 (negro)
np.min(x_train)
np.max(x_train)

np.min(x_test)
np.max(x_test)

# Pasamos a escala [0,1]
x_train = x_train/255
x_test = x_test/255


# Veamos las etiquetas
print(y_train[0:10])
print(y_test[0:10])

# Codificacion one-hot con la funcion to_categorical()
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(y_train[0:10,])
print(y_test[0:10,])

############################
## Red neuronal multicapa ## 
############################

modelo = Sequential([
        Input(shape=(28,28,1)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
])

# Veamos los detalles del modelo
modelo.summary()

# Compilacion

modelo.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = modelo.fit(x_train, y_train, 
                     epochs = 10, 
                     batch_size = 512,
                     validation_split = 0.2)  

# Visualizmos el resultado
print(history.history.keys())
df = pd.DataFrame(history.history)
print(df)
df.plot()

# Representamos por separado la evolución de la función de pérdida y el accuracy
dfAccuracy = df.loc[:,["accuracy","val_accuracy"]]
dfAccuracy.plot()

dfLoss = df.loc[:,["loss","val_loss"]]
dfLoss.plot()

# Verificamos si el modelo tambien funciona bien en el conjunto de prueba
metrics = modelo.evaluate(x_test, y_test)

# Prediccion
predicciones = modelo.predict(x_test)
# Obtenemos la categoría donde se produce el máximo de la probabilidad de clasificación
predic_test = np.argmax(predicciones, axis=1)

# Categorias en test
originales_test = np.argmax(y_test, axis=1)

# Matriz de contingencia o matriz de confusión
mc = confusion_matrix(originales_test, predic_test)
print(mc)

# Graficamos la matriz
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
disp = ConfusionMatrixDisplay(confusion_matrix = mc, display_labels = class_names)
disp = disp.plot(cmap=plt.cm.Blues,values_format='g')

# Guardamos el modelo
modelo.save('modeloPractica6.keras')

