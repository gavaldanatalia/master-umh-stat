
# Carga de datos
path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Prueba/baloncesto.csv"
baloncesto<- read.csv(path, header = T,sep = ";")

str(baloncesto)
summary(baloncesto)

library(FactoMineR)
library(dplyr)


##########################
####### Apartado 9 ####### 
##########################

# Correlaciones y grafico
R <- cor(baloncesto[, 3:14])
corrplot(R, method = "square")

# Barlett (verificar si es adecuado)
# Según el test, no es adecuado hacer PCA para este ejercicio
# pvalor>0.05, lo que significa que las correlaciones no son significativas
cortest.bartlett(R, n=10) 

# KMO (segundo criterio para verificar si es adecuado)
# valor de 0.6, adecuación baja pero puede considerarse aceptable
KMO(R)

# Debido a que el test de barlett nos ha devuelto que las correlaciones no son 
# significativas, pienso que es mejor utilizar la matriz de covarianzas.
# Si las correlaciones no son significativas, puede que el PCA no represente bien 
# la variabilidad de las variables.

# Como conclusión si hacer o no PCA, pues parece que no es el mejor dataset para
# aplicar esta técnica de reducción de la dimensionalidad.

##########################
### Apartado 10 y 11 #####
##########################

# Hacemos PCA
pca_1 <- prcomp(baloncesto[, 3:14], scale = T)

# Valores propios
pca_1$sdev^2

# Contribución de cada variable a la PC1
fviz_contrib(pca_1, choice = "var", axes = 1, top = 10)

# Las variables con mayor importancia en la componente principal 1 son rebores y porc_t2, 
# ambas con más de un 15% de importancia. Por otro lado, tenemos la variable val y tap_con
# que apenas se explican por la PC1, tap_con no llega ni a un 5%.ç

# Según la importancia de las componentes, parece que elegir 4 componentes es adeacudo,
# pues se explica un poco más del 80% de la varianza. Por otro lado, gráficamente podemos ver
# un codo a partir de la 3 componente, pero elegiriamos 4 componentes ya que llegamos a un %
# de varianza explicada superrior al 80%
# Según el valor de los valores propios, nos quedariamos entre 3 y 4. El cuarto valor propio
# es practicamente 1 (después del cuarto, sus valores son menores a 1)

# % varianza explicada
summary(pca_1)
plot(pca_1)

# Gráfico de codo
fviz_screeplot(pca_1)
fviz_screeplot(pca_1, addlabels= T) # codo en PC3

# Conclusión: 4 componentes prinicpales porque la varianza es mayor del 80%

##########################
####### Apartado 12 ######
##########################

# Extraemos las cargas (loadings) de las variables
loadings <- as.data.frame(pca_1$rotation[, 1:3])  # Tomamos PCA1, PCA2 y PCA3
loadings$Variable <- rownames(loadings)          # Añadimos los nombres de las variables
loadings

# Esto puede ocurrir porque la variable val esta correlacionada negativamente con la PCA1
# de manera que cuando la PCA toma valores altos, esta tiene normalmente valores bajos

##########################
####### Apartado 13 ######
##########################

# Normalizamos los datos (escalado para igualar las unidades de medida)
baloncesto <- scale(baloncesto[, 3:14])
baloncesto <- as.data.frame(baloncesto)  # Convertimos a data.frame para facilitar su manejo
describe(baloncesto)

# Grafico de elección de k
# No se ve demasiado claro, parece que puede estar entre 2 y 3 el codo
fviz_nbclust(baloncesto, kmeans, method = "wss")

# metodo de silueta: nos indica que es 2 el número optimo
fviz_nbclust(baloncesto, kmeans, method = "silhouette")

# Ante la duda de ambos métodos, nos quedamos con k=2
set.seed(321) 
k2 <- kmeans(baloncesto, centers = 2, nstart = 25)
k2

##########################
####### Apartado 14 ######
##########################

fviz_cluster(k2, data = baloncesto)

# Se ven dos grupos diferenciados, parece que el grupo 2 toma valores más bajos en la dim1
# en cambio, toma valores más altos en la dim 2. En el caso del grupo 1 (rojo) es al contrario.
# Existen observaciones que colindan casi entre el perímetro de ambos clusters, es lo 
# que se aprecia en la observación 7 y 2.

##########################
####### Apartado 15 ######
##########################

baloncesto %>% 
  mutate(cluster = k2$cluster) %>%  # Añadimos la asignación de clústeres
  group_by(cluster) %>%  # Agrupamos por clúster
  summarise_all("mean")  # Calculamos las medias de cada variable por clúster

# El cluster 1 toma valores más bajos, en media, en las variables val, porc_t2, porc_t1, 
# rebotes, asist, bal_rec,bal_per, tap_fav. En cambio, en la variable de porc_t3 son más altos.
# En cambio, el cluster 2 toma valores más bajos en las variables que el cluster 1 toma más altos. 
# Pasa igualmente al contrario con la variable de porc_t3.


##########################
####### Apartado 16 ######
##########################

d <- get_dist(baloncesto, method = "euclidean")
hc_diana <- diana(x = d, diss = TRUE, stand = FALSE)

fviz_dend(x = hc_diana, cex = 0.5) +
  labs(title = "Hierarchical clustering divisivo",
       subtitle = "Distancia euclídea")

# Parece que un corte en 6 clusters podría ser adecuado.
# Cortariamos por el valor 7.5

fviz_dend(hc_diana, cex = 0.8, lwd = 0.8, k = 3,
          k_colors = c("red", "green3", "blue", "magenta"),
          rect = TRUE, 
          rect_border = "gray", 
          rect_fill = FALSE)


# La interpretación es que en el primer cluster estarían las obs 1 y 5
# en el segundo las obs 15, 19, 16 y 17. El resto de obs pertenecerian al cluster azul.
