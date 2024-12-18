library(ggplot2)
library(dplyr)
library(mvoutlier)

path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Parte 3 - Scripts docente/9_DatosAtipicosDatosFaltantes/ejercicio2-datosAtipicos/StudentsPerformance.csv"

StudentsPerformance<- read.csv(path, header = T,sep = ",")

str(StudentsPerformance)

#APARTADO 1. 1.	Muestra la estructura de la base de datos y realiza un resumen 
# descriptivo de los datos. De la base de datos, aquellas variables declaradas 
# como enteras, cámbialas a  numéricas. Visualiza mediante histogramas cada 
# una de distribuciones de las variables “writing.score”, “reading.score” y “math.score”.

data <- StudentsPerformance %>% 
  select(writing.score, reading.score,math.score) %>% 
  mutate(writing.score = as.numeric(writing.score),
         reading.score = as.numeric(reading.score),
         math.score = as.numeric(math.score))

str(data)

#Analizamos la distribución
summary(data)

library(patchwork)

h1 <- ggplot(data, aes(x = writing.score)) +
  geom_histogram(bins = 10)

h2 <- ggplot(data, aes(x = reading.score)) +
  geom_histogram(bins = 10)

h3 <- ggplot(data, aes(x = math.score)) +
  geom_histogram(bins = 10)

hist <- (h1 | h2 | h3); hist

#APARTADO 2.	Visualiza la distribución de la   variable “writing.score” mediante
#un gráfico de cajas. Indica si el gráfico muestra algún outlier. Si es así, 
# indica qué valores son y en qué posición se encuentran.

ggplot(data, aes(y = writing.score)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal()

#identificamos 5 valores atípicos
boxplot.stats(data$writing.score)

boxplot.stats(data$writing.score)$out

#localizamos la posición de los datos 
which(data$writing.score %in% boxplot.stats(data$writing.score)$out)

#APARTADO 3. Realiza un gráfico de dispersión de la variable “writing.score” 
# (variable x debe ser index) en el que se muestre una línea divisoria en color
# rojo a 3 desviaciones típicas de la media que visualice los posibles outliers.

#gráfico de dispersión
# Para este ejemplo utilizamos 3 desviaciones típicas

outliers_max<-mean(data$writing.score)+3*sd(data$writing.score)
outliers_max
outliers_min<-mean(data$writing.score)-3*sd(data$writing.score)
outliers_min

plot(data$writing.score, main="Diagrama de Dispersión")
abline(h=c(outliers_max,outliers_min), col="red",lty=5)


#APARTADO 4. Realiza un gráfico de densidad de la variable “writing.score” 
# en el que se muestre líneas verticales a 3 desviaciones típicas de la media.

ggplot(data, aes(x = writing.score)) +
  geom_density()+
  geom_vline(xintercept = c((mean(data$writing.score) - 3*sd(data$writing.score)),
                            (mean(data$writing.score) + 3*sd(data$writing.score))),
             linetype="dotted",color="blue") + #agregamos lineas verticales, para identificar valores outliers
  theme_minimal()

# APARTADO 5. Realiza el test de Grubbs y el test de Rosner para los 
# potenciales outliers de la variable “writing.score”. Interpreta los 
# resultados obtenidos.


#Test de grubbs
#Existe al menos un valor atípico en los datos.
#Si el p-valor es menor que el nivel de significancia 𝛼, el test indica que hay un outlier.
#Si el p-valor es mayor que α, no hay evidencia de que el valor más extremo sea un outlier
#el término "más extremo" se refiere al valor que está más alejado de la media de los datos, 
#ya sea por ser el más grande o el más pequeño..
library(outliers)

grubbs.test(data$writing.score) #A un nivel de significación de 0.1, considera
#que el valor 10 es un outlier, si lo quitamos y realizamos el test con el restante de valores:

which(data$writing.score == 10) # encontrando el valor 10
grubbs.test(data$writing.score[-60])

which(data$writing.score == 15) # encontrando el valor 15
grubbs.test(data$writing.score[-c(597,60)])

which(data$writing.score == 19) # encontrando el valor 15
grubbs.test(data$writing.score[-c(597,60,328)])

#Prueba de Rosner
# El test de Rosner es útil para identificar múltiples outliers simultáneamente en un conjunto de datos univariados. 
# Hay k valores atípicos en los datos, donde k es el número máximo de outliers que deseas identificar.
library(EnvStats)
test <- rosnerTest(data$writing.score,alpha = 0.1,k =8)
test$all.stats
#El test de Rosner considera que no hay outliers.


#APARTADO 6. Visualiza un gráfico de dispersión con las tres variables 
# de calificaciones, interpreta el gráfico. Visualiza también gráficos de 
# cajas para cada una de las tres variables de calificaciones.


library(scatterplot3d) # Observamos los 3 exámenes

scatterplot3d(StudentsPerformance$math.score, 
              StudentsPerformance$reading.score, 
              StudentsPerformance$writing.score)

g1 <- ggplot(StudentsPerformance, aes(y = writing.score)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal()

g2 <- ggplot(StudentsPerformance, aes(y = math.score)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal()

g3 <- ggplot(StudentsPerformance, aes(y = reading.score)) +
  geom_boxplot(outlier.colour = "red") +
  theme_minimal()

graf <- (g1 | g2| g3); graf

#Apartado 7. Teniendo en cuenta las tres variables de calificaciones, 
# calcula la distancia de Mahalanobis y muestra los 5 valores más altos. 
data <- StudentsPerformance %>% 
  select(writing.score,math.score,reading.score)

vector_medias = colMeans(data) 
matriz_var_cov = cov(data)

# Creamos una variable con la distancia
data$maha2 = mahalanobis(data,vector_medias,matriz_var_cov)
head(data)

top_maha2 <- data %>%
  top_n(5, maha2) %>% 
  arrange(desc(maha2)) %>%
  print()
head(top_maha2)

#APARTADO 8. 	Representa en un gráfico bidimensional, haciendo uso de 
# las variables “math.score” y “writing.score”, la distancia mahalanobis 
# robusta con ajuste, utilizando diferentes colores según las distancias
# euclidianas de las observaciones. ¿Los gráficos representados indican 
# algún outlier?

library(mvoutlier)

Z <- cbind(data$writing.score,data$math.score)

color.plot(Z)
color.plot(Z)$outlier

which(color.plot(Z)$outlier == TRUE)


library(dplyr)

Z %>% 
  as.data.frame() %>% 
  slice(c(18,60,77,92,146,212,328,339,340,364,467,529,597,602,788,843,897,981)) %>% 
  rename(writing.score = "V1",  math.score ="V2")

res <- aq.plot(Z)

