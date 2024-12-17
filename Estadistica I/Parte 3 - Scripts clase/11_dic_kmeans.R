
install.packages("factoextra")
library(dplyr)
library(psych)
library(factoextra)

#APARTADO 1
data("USArrests")
datos = USArrests
head(datos)
str(datos)


describe(datos)

datos = scale(datos)
describe(datos)

datos = as.data.frame(datos)

#Calcular la matriz de distancias y representarlas
matriz = get_dist(datos, method="euclidean")
fviz_dist(matriz)

#Apartado 3

##codo
fviz_nbclust(datos, kmeans, method="wss")

##silueta
fviz_nbclust(datos, kmeans, method="silhouette")

#Apartado 4
set.seed(1234)

k2 = kmeans(datos, centers = 2, nstart = 25)

#centroides
k2$centers

#centroides (datos originales)
USArrests %>%
  mutate(cluster = k2$cluster) %>%
  group_by(cluster) %>%
  summarise_all("mean")

# apartado 5
fviz_cluster(k2, 
             data=datos,
             repel = TRUE,
             star.plot = TRUE)

# apartado 6
# Apartado 6
library(tidyverse)
datos$clus <- as.factor(k2$cluster)
data_long <- datos %>%
  pivot_longer(cols = Murder:Rape,
               names_to = "var",
               values_to = "valor")
ggplot(data_long, aes(x=var, y=valor, group=clus, color = clus)) +
  geom_point() + 
  stat_summary(fun = mean, geom = "pointrange", size = 1)



