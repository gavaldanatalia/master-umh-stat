# Análsis factorial

# Instalar los paquetes necesarios
if (!require("psych")) install.packages("psych")
if (!require("GPArotation")) install.packages("GPArotation")

# Cargar los paquetes
library(psych)
library(GPArotation)

##### APARTADO 1
# Cargar los datos
path="/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Parte 3 - Scripts docente/11_AnalisisFactorial/Ejemplos/EjemploCuestionario/cuestionario.csv"
data <- read.csv(path, sep = ";")
str(data)

# Visualizar los primeros datos
head(data)

# Número de observaciones
n <- nrow(data)


##### APARTADO 2
# Test de esfericidad de Bartlett
cortest.bartlett(cor(data[, 1:25]), n = n)

# Dado que el p-valor es extremadamente bajo:
# Las correlaciones entre las variables son estadísticamente significativas.
# Es adecuado proceder con el análisis factorial, ya que existe suficiente correlación 
# entre las variables para justificarlo.


# Test de adecuación muestral (KMO)
KMO(cor(data[, 1:25]))

# La salida del test de adecuación muestral de Kaiser-Meyer-Olkin (KMO) 
# evalúa si las variables de tu conjunto de datos son adecuadas para realizar un análisis factorial
# Según los criterios establecidos:
# > 0.90: Excelente.
# 0.80 - 0.89: Muy bueno.
# 0.70 - 0.79: Bueno.
# 0.60 - 0.69: Aceptable.
# < 0.60: Inadecuado.

# MSA (Measure of Sampling Adequacy) evalúa la adecuación de cada variable
# < 0.50: La variable no es adecuada y debería eliminarse.
# > 0.50: La variable es adecuada.

##### APARTADO 3
# Determinar el número de factores (scree plot) - "punto de codo"
fa.parallel(data[, 1:25], fa = "both", n.iter = 100, show.legend = TRUE)

# Cuando los valores empiezan a estabilizarse, cerca del 1
eigenvalues <- eigen(cor(data[, 1:25]))$values
print(eigenvalues)

# Entre 5 y 6 factores. Seguimos con 5 factores.

##### APARTADO 4
# Realizar el análisis factorial exploratorio
fa_result <- fa(data[, 1:25], nfactors = 5, rotate = "varimax")
# Visualizar los resultados
print(fa_result, digits = 3, cutoff = 0.3)

# Si alguna comunalidad es muy baja (< 0.40), esa variable no está bien explicada por los factores.
# Variables con comunalidades bajas (< 0.40):
# Estas variables pueden eliminarse del modelo o 
# evaluarse si se agrupan con otras variables.
fa_result$communality

# Graficar la matriz de cargas factoriales
fa.diagram(fa_result)

# Cargas bajas
# Variables con cargas bajas (< 0.30):
# Estas variables no contribuyen significativamente a ningún factor 
# y podrían eliminarse.
# En este caso no hay ninguna.
low_loadings <- apply(abs(fa_result$loadings), 1, max) < 0.30
names(low_loadings[low_loadings])

# varianza explicada
# Esto muestra la varianza total explicada por los factores 
# y la proporción que explica cada factor.
# Si la varianza total explicada por los factores es baja (< 50 %), 
# podrías necesitar más factores o revisar la estructura factorial.
fa_result$Vaccounted

##### APARTADO 5
# Ajustar un modelo factorial
fa_model <- fa(data[, 1:25], nfactors = 5, rotate = "varimax", fm = "ml")

# Calcular puntuaciones factoriales usando el modelo
factor_scores <- factor.scores(data[, 1:25], fa_model, method = "regression")

# Ver las puntuaciones de 5 sujetos
head(factor_scores$scores)
length(factor_scores$scores)/5 # Sujetos totales 
dim(data)[1] # Sujetos totales en df original

# Representar a los encuestados en los ejes de los factores
# Crear un gráfico de dispersión de los factores 1 y 2
plot(
  factor_scores$scores[, 1],  # Puntuaciones del Factor 1
  factor_scores$scores[, 2],  # Puntuaciones del Factor 2
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Representación de los encuestados en los ejes factoriales",
  pch = 19, col = "blue"
)

# Agregar una cuadrícula
grid()

# Representar los factores coloreados por género
plot(
  factor_scores$scores[, 1],  # Factor 1
  factor_scores$scores[, 2],  # Factor 2
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Representación por género",
  pch = as.numeric(data$gender),  # Diferentes símbolos por género
  col = as.numeric(data$gender)  # Diferentes colores por género
)
legend("topright", legend = c("Hombres", "Mujeres"), col = c(1, 2), pch = c(1, 2))

# Cambiar los factores a graficar
plot(
  factor_scores$scores[, 3],  # Factor 3
  factor_scores$scores[, 4],  # Factor 4
  xlab = "Factor 3",
  ylab = "Factor 4",
  main = "Representación de los factores 3 y 4",
  pch = 19, col = "green"
)



##### APARTADO 6

# Verificar las categorías de la variable gender
table(data$gender)

# Si gender es categórica pero no numérica, conviértela en factor
data$gender <- as.factor(data$gender)

# Graficar Factor 1 vs. Factor 2 coloreando por género
plot(
  factor_scores$scores[, 1],  # Puntuaciones del Factor 1
  factor_scores$scores[, 2],  # Puntuaciones del Factor 2
  col = as.numeric(data$gender),  # Colores según el género
  pch = 19,  # Forma de los puntos
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Gráfico de factores diferenciados por género"
)
legend(
  "topright",
  legend = levels(data$gender),  # Etiquetas de género
  col = 1:length(levels(data$gender)),  # Colores según género
  pch = 19  # Forma de los puntos
)

# Promedios de los factores por género
aggregate(factor_scores$scores, by = list(Gender = data$gender), FUN = mean)

# Desviaciones estándar de los factores por género
aggregate(factor_scores$scores, by = list(Gender = data$gender), FUN = sd)


##### APARTADO 7
# Verificar las categorías de la variable education
table(data$education)

# Si education es categórica pero no numérica, conviértela en factor
data$education <- as.factor(data$education)

# Graficar Factor 1 vs. Factor 2 coloreando por género
plot(
  factor_scores$scores[, 1],  # Puntuaciones del Factor 1
  factor_scores$scores[, 2],  # Puntuaciones del Factor 2
  col = as.numeric(data$education),  # Colores según el género
  pch = 19,  # Forma de los puntos
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Gráfico de factores diferenciados por género"
)
legend(
  "topright",
  legend = levels(data$education),  # Etiquetas de género
  col = 1:length(levels(data$education)),  # Colores según género
  pch = 19  # Forma de los puntos
)

# Promedios de los factores por género
aggregate(factor_scores$scores, by = list(Gender = data$education), FUN = mean)

# Desviaciones estándar de los factores por género
aggregate(factor_scores$scores, by = list(Gender = data$education), FUN = sd)


##### APARTADO 8
# Categorizar la edad en tres grupos
data$age_group <- cut(
  data$age,
  breaks = c(0, 30, 60, Inf),  # Intervalos: Menores de 30, 30-60, Mayores de 60
  labels = c("Joven", "Adulto", "Mayor")  # Nombres de las categorías
)

# Verificar la distribución de las categorías
table(data$age_group)

# Graficar Factor 1 vs. Factor 2 coloreando por categoría de edad
plot(
  factor_scores$scores[, 1],  # Puntuaciones del Factor 1
  factor_scores$scores[, 2],  # Puntuaciones del Factor 2
  col = as.numeric(data$age_group),  # Colores según la categoría de edad
  pch = 19,  # Forma de los puntos
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Gráfico de factores diferenciados por edad"
)

# Agregar leyenda
legend(
  "topright",
  legend = levels(data$age_group),  # Etiquetas de las categorías de edad
  col = 1:length(levels(data$age_group)),  # Colores según categoría
  pch = 19  # Forma de los puntos
)

# Graficar Factor 3 vs. Factor 4 coloreando por categoría de edad
plot(
  factor_scores$scores[, 3],  # Factor 3
  factor_scores$scores[, 4],  # Factor 4
  col = as.numeric(data$age_group),  # Colores según la categoría de edad
  pch = 19,  # Forma de los puntos
  xlab = "Factor 3",
  ylab = "Factor 4",
  main = "Gráfico de factores 3 y 4 diferenciados por edad"
)

# Agregar leyenda
legend(
  "topright",
  legend = levels(data$age_group),  # Etiquetas de las categorías de edad
  col = 1:length(levels(data$age_group)),  # Colores según categoría
  pch = 19  # Forma de los puntos
)

# Promedios de las puntuaciones factoriales por categoría de edad
aggregate(factor_scores$scores, by = list(AgeGroup = data$age_group), FUN = mean)

# Desviaciones estándar de las puntuaciones factoriales por categoría de edad
aggregate(factor_scores$scores, by = list(AgeGroup = data$age_group), FUN = sd)


