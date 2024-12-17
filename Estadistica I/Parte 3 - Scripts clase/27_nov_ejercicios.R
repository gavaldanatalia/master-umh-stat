# Estadistica 1

path='/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/StudentsPerformance.csv'
datos<-read.table(path, 
                     header=T, sep=",")

# PDF con codigo en el campus virtual

#install.packages("outliers")
library(outliers)

# Vemos cuales son los outliers
grubbs.test(datos$writing.score)

# Quitamos la observación 70
grubbs.test(datos$writing.score[-60])

# Scatterplot

# install.packages("scatterplot3d")
library(scatterplot3d)

scatterplot3d(
  datos$math.score,
  datos$writing.score,
  datos$reading.score
)

g1 <- ggplot(datos,aes(y=writing.score)) +
  geom_boxplot(outlier.colour = 'red') +
  theme_minimal()
g2 <- ggplot(datos,aes(y=math.score)) +
  geom_boxplot(outlier.colour = 'red') +
  theme_minimal()
g3 <- ggplot(datos,aes(y=reading.score)) +
  geom_boxplot(outlier.colour = 'red') +
  theme_minimal()

graf <- (g1 | g2 | g3); graf

# Ejercicio 7

data <- datos %>%
  select(writing.score, math.score, reading.score)

vector_medias = colMeans(data)
matriz_var_cov = cov(data)

data$maha2 = mahalanobis(data, vector_medias, matriz_var_cov)
head(data)

# Top maha2
top <- data %>%
  top_n(5, maha2) %>%
  arrange(desc(maha2)) %>%
  print()

# Ejercicio 8

# install.packages("mvoutlier")
library(sgeostat)
library(mvoutlier)

Z <- cbind(mpg$cty, mpg$hwy)
color.plot(Z)
