# Cargar librerías necesarias
library(fpc)
library(factoextra)
library(dbscan)
library(ggplot2)

# Cargar y escalar los datos
data("DS3")
datos <- DS3
datos <- scale(datos)

# Visualizar los datos originales
plot(datos, pch = 20, cex = 0.25, main = "Datos Originales")

# -----------------------
# 1. Clustering K-means
# -----------------------
set.seed(321)
km_clusters <- kmeans(x = datos, centers = 6, nstart = 50)

# Visualización de K-means
fviz_cluster(object = km_clusters, data = datos, geom = "point", ellipse = FALSE,
             show.clust.cent = FALSE) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Clustering K-means")

# -----------------------
# 2. Clustering DBSCAN
# -----------------------
dbscan_cluster <- fpc::dbscan(data = datos, eps = 0.1, MinPts = 22)

# Visualización de DBSCAN
fviz_cluster(object = dbscan_cluster, data = datos, stand = FALSE,
             geom = "point", ellipse = FALSE, show.clust.cent = FALSE) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Clustering DBSCAN")

# -----------------------
# 3. Clustering Jerárquico
# -----------------------

# Calcular la matriz de distancias y aplicar complete linkage
dist_matrix <- dist(datos)
hc_complete <- hclust(dist_matrix, method = "complete")

# Crear los clústeres (elegir el número deseado, por ejemplo, 6)
hc_clusters <- cutree(hc_complete, k = 6)

# Visualización de los clusters jerárquicos en el espacio
fviz_cluster(list(data = datos, cluster = hc_clusters), geom = "point", ellipse = FALSE,
             show.clust.cent = FALSE) +
  theme_bw() +
  theme(legend.position = "none") +
  ggtitle("Clustering Jerárquico (Complete Linkage)")



