# APARTADO 1: Cargar y explorar los datos
# Cargamos el dataset USArrests que contiene tasas de crímenes en los estados de EE.UU.
data("USArrests")
datos <- USArrests

# Inspeccionamos las primeras filas del dataset y su estructura
head(datos)
str(datos)

# Calculamos estadísticas descriptivas para las variables
library(psych)
describe(datos)

# Normalizamos los datos (escalado para igualar las unidades de medida)
datos <- scale(datos)
datos <- as.data.frame(datos)  # Convertimos a data.frame para facilitar su manejo
describe(datos)  # Volvemos a calcular estadísticas descriptivas para verificar el escalado

# APARTADO 2: Matriz de distancias y visualización
# Calculamos la matriz de distancias euclidianas entre observaciones
library(factoextra)
matriz.dis <- get_dist(datos, method = "euclidean")

# Visualizamos la matriz de distancias en un gráfico
fviz_dist(matriz.dis)

# APARTADO 3: Determinación del número óptimo de clústeres
# Método del codo (wss): Evalúa la inercia dentro de los clústeres
fviz_nbclust(datos, kmeans, method = "wss")

# Método de la silueta: Evalúa la separación y compacidad de los clústeres
fviz_nbclust(datos, kmeans, method = "silhouette")

# APARTADO 4: Aplicación de K-means
# Ejecutamos el algoritmo K-means con 2 clústeres (basado en análisis previo)
set.seed(1234)  # Fijamos semilla para reproducibilidad
k2 <- kmeans(datos, centers = 2, nstart = 25)
k2  # Resumen de los resultados

# Mostramos los centroides de los clústeres escalados
k2$centers

# Calculamos los centroides de las variables sin escalar
library(dplyr)
USArrests %>% 
  mutate(cluster = k2$cluster) %>%  # Añadimos la asignación de clústeres
  group_by(cluster) %>%  # Agrupamos por clúster
  summarise_all("mean")  # Calculamos las medias de cada variable por clúster

# APARTADO 5: Visualización de los clústeres
# Representamos gráficamente los clústeres en el espacio de datos
fviz_cluster(k2, data = datos)

# Visualización mejorada con elipses euclidianas, etiquetas desplazadas y líneas hacia los centroides
fviz_cluster(k2, data = datos, ellipse.type = "euclid",
             repel = TRUE, star.plot = TRUE)

# APARTADO 6: Visualización de medias por clúster
# Transformamos los datos a formato largo para facilitar la comparación
datos$clus <- as.factor(k2$cluster)  # Añadimos la asignación de clústeres como factor
library(tidyverse)
data_long <- datos %>%
  pivot_longer(cols = Murder:Rape, names_to = "variable", values_to = "valor")

# Graficamos las medias de las variables por clúster
ggplot(data_long, aes(x = variable, y = valor, group = clus, color = clus)) +
  stat_summary(fun = mean, geom = "pointrange", size = 1) +  # Media con barras de error
  geom_point(size = 1)  # Puntos individuales para mayor claridad
