# Carga de datos
library(mvoutlier)

path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Prueba/vino.csv"
vino<- read.csv(path, header = T,sep = ";")

# Vemos el número de datos observaciones que tenemos
# Disponemos de 4 variables.
str(vino)

##########################
####### Apartado 1 ####### 
##########################
# Gráfico de cajas de la variable density
ggplot(vino, aes(y = density)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal() # 1 outlier

# Gráfico de cajas de la variable ph
ggplot(vino, aes(y = ph)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal() # 3 outlier

# Gráfico de cajas de la variable sulphates
ggplot(vino, aes(y = sulphates)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal() # 6 outlier

# Parece que la variable que mas outlier tiene es la de sulphates con 6
# outlier, mientras que ph y density tienen menos outlier. Eso siginica que
# para una 6 vinos, los sulphates son más extremos de lo esperado.

##########################
####### Apartado 2 ####### 
##########################

library(EnvStats)

# Ho: No hay valores atípicos en la variable X
# H1: Hay al menos un valor atípico en la variable X

# k=1 porque solo apreciamos 1 outlier gráficamente
# significancia: 10%
# El test devuelve que efectivamente es un outlier (se cofirma lo que vemos graficamente)
# Observación 193
test_density <- rosnerTest(vino$density, alpha = 0.1, k=1) 
test_density$all.stats

# k=3 porque solo apreciamos 3 outlier gráficamente
# significancia: 10%
# El test devuelve que no son outliers (NO se cofirma lo que vemos graficamente)
test_ph <- rosnerTest(vino$ph, alpha = 0.1, k=3) 
test_ph$all.stats

# k=6 porque solo apreciamos 6 outlier gráficamente
# significancia: 10%
# El test devuelve que no son outliers (NO se cofirma lo que vemos graficamente)
# salvo uno de los puntos, el más alejado. La observación número 274
test_sulphates <- rosnerTest(vino$sulphates, alpha = 0.1, k=6) 
test_sulphates$all.stats

# En resumen, con el test de Rosner, tenemos un total de 2 outlier
# un outlier en la variable de density y otro en la de sulphates
# Observaciones consideradas outlier --> 193 y 274


##########################
####### Apartado 3 ####### 
##########################

# Seleccionamos únicamente las variables numéricas
vino_num <- vino %>% 
  select(density, ph, sulphates)

# Calculamos el vector de medias por variable y las covarianzas
vector_medias = colMeans(vino_num) 
matriz_var_cov = cov(vino_num)

# Creamos una columna con la distancia
vino_num$maha2 = mahalanobis(vino_num,vector_medias,matriz_var_cov)
head(vino_num)

# Vemos las distancias en un df
top_maha2 <- vino_num %>%
  top_n(5, maha2) %>% 
  arrange(desc(maha2)) %>%
  print()
head(top_maha2)

# Grafico tridimensional con scatterplot3d
library(scatterplot3d) 

scatterplot3d(vino_num$ph, 
              vino_num$density, 
              vino_num$sulphates)

# Vemos cual es la observación con esa densidad y vemos que es la 193 (antes detectada como outlier)
which(vino_num$density == 1.03898, arr.ind = TRUE)

# Observación
vino_num[193,]

##########################
####### Apartado 4 ####### 
##########################

# Unimos las variables que tienen outliers y hacemos el gráfico
# Vemos que el valor marcado en rojo, así como otros que exceden del gráfico
# son outliers
Z <- cbind(vino_num$density,vino_num$sulphates)
color.plot(Z)

# Como comentabamos anteriormente, la obs 193 es un outlier, es la 
# que aparece en el gráfico de color rojo
vino_num[193,]

# Otros outliers que categoriza la función
# corresponden a aquellos puntos que salen de los círculos
which(color.plot(Z)$outlier == TRUE)

