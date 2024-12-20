library(cluster)
library(purrr)
library(factoextra)
library(tidyverse)
library(dplyr)

data("USArrests")
datos <- USArrests
head(datos)
str(datos)
dim(datos)
datos <- scale(datos)
datos <- as.data.frame(datos)
d <- get_dist(datos, method = "euclidean")

# Hierarchical clustering using Complete Linkage (máximo)
hc1 <- hclust(d, method = "complete" )
hc1
# Plot the obtained dendrogram
fviz_dend(hc1, rect = TRUE, cex = 0.5)

# Hierarchical clustering using Linkage (mínimo)
hc2 <- hclust(d, method = "single" )
fviz_dend(hc2, rect = TRUE, cex = 0.5)

# Hierarchical clustering using Linkage (Media)
hc3 <- hclust(d, method = "average" )
fviz_dend(hc3, rect = TRUE, cex = 0.5)

# Hierarchical clustering using Linkage (centroid)
hc4 <- hclust(d, method = "centroid")
hc4
fviz_dend(hc4, rect = TRUE, cex = 0.5)

# Hierarchical clustering using Linkage (Ward)
hc5 <- hclust(d, method = "ward.D2" )
fviz_dend(hc5, rect = TRUE, cex = 0.5)


# La selección del número óptimo puede valorarse de forma visual, 
# tratando de identificar las ramas principales en base a la altura a la que ocurren las uniones. 
# En el ejemplo expuesto es razonable elegir entre 2 o 4 clusters.
# Elegimos 4 clústers, Se pueden añadir colores
library(parameters)
library(mclust)
library(NbClust)
library(see)
library(igraph)

n_clust <- n_clusters(datos,
                      nbclust_method = "ward.D2",
                      package = c("easystats", "NbClust", "mclust"),
                      standardize = FALSE)
n_clust
plot(n_clust)
as.data.frame(n_clust)

# En el ejemplo expuesto es razonable elegir entre 2 o 4 clusters. 
# (fijarse en las barras más altas)
# En kmedias elegiomos 2 clústers, en este vamos a ver como resultan 4 clústers
# Elegimos 4 clústers, Se pueden añadir colores
# También se puede utilizar alguno de los métodos vistos en kmedias
fviz_dend(hc1, cex = 0.8, lwd = 0.8, k = 4,
          k_colors = c("red", "green3", "blue", "magenta"),
          rect = TRUE, 
          rect_border = "gray", 
          rect_fill = FALSE)

#Se puede elegir si vertical u horizontal
fviz_dend(hc1, cex = 0.8, lwd = 0.8, k = 4,
          k_colors = c("red", "green3", "blue", "magenta"),
          rect = TRUE, 
          rect_border = "gray", 
          rect_fill = FALSE, 
          horiz = TRUE)

#Se puede elegir si tipo de dendrograma
fviz_dend(hc1, cex = 0.8, lwd = 0.8, k = 4,
          k_colors = c("red", "green3", "blue", "magenta"),
          rect = TRUE, 
          rect_border = "gray", 
          rect_fill = FALSE, 
          type= "circular")

fviz_dend(hc5, cex = 0.8, lwd = 0.8, k = 4,
          k_colors = c("red", "green3", "blue", "magenta"),
          rect = TRUE, 
          rect_border = "gray", 
          rect_fill = FALSE, 
          type= "phylogenic")

# Representando en el plano, es como realizar primero un CP y luego
# representar a los estados en ambos
clust <- cutree(hc5, k = 4)
clust
fviz_cluster(list(data = datos, cluster = clust), labelsize = 6)

# Representación de las medias de cada variable en cada clúster con 
# las variables estandarizadas.
cluster_paises <- datos
cluster_paises$clust <- as.factor(clust)
data_long <- cluster_paises %>%
  pivot_longer(cols = Murder:Rape, names_to = "variable", values_to = "valor")
ggplot(data_long, aes(as.factor(variable), y =valor, group = clust, colour = clust))+
  stat_summary(fun = mean, geom = "pointrange", size = 1)+
  stat_summary(geom = "line")+
  geom_point(aes(shape = clust))

#Obtenemos las medias de cada grupo, sin estandarizar
USArrests %>% 
  mutate(cluster = cluster_paises$clust) %>% 
  group_by(cluster) %>% 
  summarise_all("mean")

##########################################
#Análisis de Conglomerados Divisivo (DIANA)

hc_diana <- diana(x = d, diss = TRUE, stand = FALSE)

fviz_dend(x = hc_diana, cex = 0.5) +
  labs(title = "Hierarchical clustering divisivo",
       subtitle = "Distancia euclídea")

##########################################
#Análisis de Conglomerados FUZZY (Fanny)
datos <- scale(USArrests)

library(cluster)
fuzzy_cluster <- fanny(x = datos, diss = FALSE, k = 4, 
                       metric = "euclidean",
                       stand = FALSE)

# El objeto devuelto fanny() incluye entre sus elementos: una matriz con el grado de pertenencia de cada observación a cada cluster 
# (las columnas son los clusters y las filas las observaciones).
head(fuzzy_cluster$membership)

# El coeficiente de partición Dunn normalizado y sin normalizar. Valores normalizados próximos 
# a 0 indican que la estructura tiene un alto nivel fuzzy 
# y valores próximos a 1 lo contrario.
fuzzy_cluster$coeff

# El cluster al que se ha asignado mayoritariamente cada observación.
head(fuzzy_cluster$clustering)
fviz_cluster(object = fuzzy_cluster, repel = TRUE, ellipse.type = "norm",
             pallete = "jco") + theme_bw() + labs(title = "Fuzzy Cluster plot")


#####
library(clValid)
comparacion <- clValid(
  obj        = datos,
  nClust     = 2:4,
  clMethods  = c("hierarchical", "kmeans", "diana","fanny"),
  validation = c("stability", "internal")
)
summary(comparacion)
