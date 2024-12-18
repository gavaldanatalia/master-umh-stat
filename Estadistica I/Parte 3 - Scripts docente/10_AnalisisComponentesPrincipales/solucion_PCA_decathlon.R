####### EJEMPLO COMPONENTES PRINCIPALES DECATHLON########

library(FactoMineR)
library(dplyr)

# El conjunto de datos está formado por 41 filas y 13 columnas.
# Las columnas 1 a 12 son variables continuas: las primeras diez columnas 
# corresponden al desempeño de los atletas para los 10 eventos del decatlón 
# y las columnas 11 y 12 corresponden respectivamente al rango y los
# puntos obtenidos. La última columna es una variable categórica 
# correspondiente a la reunión atlética (2004# Olympic Game o 2004 Decastar).

#Apartado 1: Resumen descriptivo y análisis gráfico

data(decathlon)
str(decathlon)
summary(decathlon)
R <- cor(decathlon[, 1:10]); R

library(corrplot)
corrplot(R, method = "square")

#Apartado 2: Realizamos test DE ESFERICIDAD DE BARLETT

### KMO Y BARLETT TEST
library(psych)

cortest.bartlett(R,n=41) 
# Como p.value < 0.05, se rechaza H0, 
# lo que significa que las correlaciones 
# entre variables son significativas y el PCA es apropiado 
# para el conjunto de datos

KMO(R)

# Con un valor de 0.6, el índice KMO indica 
# que la adecuación para aplicar
# el PCA es baja, pero aceptable.


#APARTADO 3. Obtención de las componentes principales con la función prcomp()

pca_1 <- prcomp(decathlon[,1:10], scale = T)
# por defecto las variables se centran
# pero indicando scale=T hace el componentes principales 
# con las variables tipificadas

# vectores propios, matriz A
pca_1$rotation

# % varianza explicada
summary(pca_1)
plot(pca_1)

# Valores propios
pca_1$sdev^2

#APARTADO 4. Elección del número de componentes

#Scree plot
library(ggplot2)
library(factoextra)

fviz_screeplot(pca_1)
fviz_screeplot(pca_1, addlabels= T) # codo en PC5

# % de variaza acumulada superior a un umbral 
summary(pca_1) # superior a 80% en PC5

# Valores propios menor que un umbral
pca_1$sdev^2 # menor que 1 a partir de PC4


#APARTADO 5: 
# Obtención de las correlaciones entre componentes y variables. 
# Interpretación.

#Accedemos a los vectores de las puntuaciones de las observaciones en cada componente (Z's):
pca_1$x

#vectores propios, matriz A
pca_1$rotation

# Para la interpretación de las componentes, necesitamos 
# las correlaciones de las componentes con las variables. 
pca_2 <- PCA(X = decathlon[ ,1:10], 
             scale.unit = T, 
             ncp = 10, 
             graph = T)


##APARTADO 6: ¿Qué variables contribuyen más?

# las variables que más contribuyen - graficamente

fviz_contrib(pca_1, choice = "var", axes = 1, top = 10)
fviz_contrib(pca_1, choice = "var", axes = 2, top = 10)
fviz_contrib(pca_1, choice = "var", axes = 3, top = 10)
fviz_contrib(pca_1, choice = "var", axes = 4, top = 10)
fviz_contrib(pca_1, choice = "var", axes = 5, top = 10)

# las variables que más contribuyen - porcentajes
var <- get_pca_var(pca_1)
var$contrib # % de contribución de las variables


### APARTADO 8: Estudio individual (estudio de atletas): 
# dos atletas estarán cerca uno del otro si sus resultados de los
# eventos son cercanos. Queremos ver la variabilidad entre los individuos. 
# ¿Hay similitudes entre los individuos para todas las variables? 
# ¿Podemos establecer diferentes perfiles de individuos? 
# ¿Podemos oponer un grupo de individuos a otro?

# Representación de las observaciones en las componentes:

fviz_pca_ind(pca_1,
             col.ind = "blue",
             axes = c(1,2),
             pointsize = 1.5,
             labelsize = 3)


fviz_pca_ind(pca_1,
             col.ind = "blue",
             axes = c(3,4),
             pointsize = 1.5,
             labelsize = 3)


#Representación de observaciones y variables:

#BIPLOT, observaciones y variables en el gráfico de componentes

biplot(pca_1, scale = 0, cex = 0.5, col = c("dodgerblue3", "deeppink3"))

# El eje inferior e izquierda scala de las puntuaciones de las observaciones, 
# el eje superior y derecho
# representan las correlaciones entre componentes y variables entre [-1, 1]. 
