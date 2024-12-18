
library(psych)
library(corrplot)


estudiantes <- read.csv("estudiantes.csv", header = T, sep = ";")

#1 .estructura de la base de datos y resumen descriptivo
str(estudiantes)

describe(estudiantes)

#Matriz de correlaciones
R <- cor(estudiantes);R

corrplot(R, method = "square")

#2. Verificamos que se puede realizar un Análisis Factorial
###KMO Y BARLETT TEST

KMO(R)
cortest.bartlett(R, n = 20)


#3. Realiza el gráfico de sedimentación y elige un número de factores inicial

####
scree(R)

eigen(R);cumsum(eigen(R)$values)/sum(diag(R))

#4. Realiza el Análisis Factorial con el número de factores elegido en el apartado anterior, utiliza diferentes
# métodos de estimación.
#Comenzamos con 2 factores y con el método de mínimo cuadrados ordinarios (minimiza los residuos), empezamos sin rotar.

###
# Con la funcion fa() podemos utilizar los metodos siguientes.
# 
# minres: minimo residuo, minimiza la matriz de residuos utilizando un procedimiento OLS (Mínimos cuadrados ordinarios)
# mle: maxima verosimilitud
# pa: metodo de ejes principales
# alpha: análisis factorial alfa como se describe en Kaiser y Coffey (1965)
# minchi: minimizará el chi cuadrado ponderado del tamaño de la muestra cuando se traten correlaciones por pares con diferente número de sujetos por par
# minrak: análisis factorial de rango mínimo

modelo1 <- fa(estudiantes, rotate = "none", nfactors = 2, fm="minres")

modelo1$e.values #Valores propios de la matriz de correlaciones

modelo1$values #Valores propios de la solución de los factores comunes

modelo1$communality#comunalidades

modelo1

# CienciasNaturales       Matematicas           Frances             Latin        Literatura 
# 0.6009843             0.9950000         0.9539002         0.6929323         0.8287368 

# el 60% CNaturales está explicada por los factores
# el 99.5% de Matemáticas
# el 95.4% de Francés
# la mejor explicada es Matemáticas y la peor Ciencias Naturales

modelo1$uniquenesses#unicidades, el porcentaje que queda por explicar

modelo1$loadings

biplot.psych(fa(estudiantes,nfactors = 2,fm="minres",rotate = "none"),pch = c(21,18))  

fa.diagram(modelo1)

#La matriz de correlaciones residuales después de aplicar el modelo factorial.
modelo1$residual 

#más cómodo
residuals(modelo1) #cuanto más cerca de cero mejor es el AF


#observamos que hay alguna variable que no está bien explicada por los factores, hacemos una 
#rotación varimax

biplot.psych(fa(estudiantes,nfactors = 2,fm="minres",rotate = "varimax"),pch = c(21,18))  

#estimamos de nuevo el modelo
modelo2 <- fa(estudiantes,nfactors = 2, fm="minres",rotate = "varimax")

modelo2$communalities#comunalidades
modelo2$uniquenesses#unicidades

modelo2$loadings

fa.diagram(modelo2)


modelo3 <- fa(estudiantes,nfactors = 2, fm="pa",rotate = "noe")#no llega a la solución, subimos el número de iteraciones

modelo3 <- fa(estudiantes,nfactors = 2, fm="pa",rotate = "none", max.iter=100)

biplot.psych(modelo3)

modelo3$communality#comunalidades
modelo3$uniquenesses#unicidades

modelo3$loadings

fa.diagram(modelo3)

residuals(modelo3)

#Realizamos una rotación ortogonal

modelo4 <- fa(estudiantes,rotate = "Varimax",nfactors = 2,fm="pa",max.iter=100)

biplot.psych(modelo4)

modelo4$loadings

fa.diagram(modelo4)

#Realizamos una rotación oblimin

modelo5<-fa(estudiantes,rotate = "oblimin",nfactors = 2,fm="pa",max.iter=100)

biplot.psych(modelo5)

#Al hacer rotación oblicua se analiza la matriz estructura

modelo5$Structure


#Soluciones adecuadas:
## Modelo 2, 4 y 5



#5. Una vez elegido el método y rotación más adecuado, calcula las puntuaciones de los sujetos
#con el modelo factorial estimado.

#Podemos calcular las puntuaciones de los sujetos en los factores.
modelo2$scores
modelo4$scores
modelo5$scores







############################################################################
#library("factoextra")


#otra forma de hacer el ejemplo

estudiantes<-read.csv("estudiantes.csv", header = T, sep = ";")

str(estudiantes)
summary(estudiantes)

R <- cor(estudiantes);R

library(corrplot)
corrplot(R, method = "square")

###KMO Y BARLETT TEST

KMO(R)
cortest.bartlett(R, n = 20)

factanal(estudiantes, factors = 2, rotation = "none") #es de la librer?a stats, al final de los resultados
#incluye una prueba chi para determinar el n?mero de factores adecuado. 
#esta prueba estad?stica me determinar? si son suficientes o no el n?mero de factores.

# UNICIDAD (Uniquenesses): Es el porcentaje de varianza que no ha sido explicada por el Factor y es igual a: 1 - Comunalidad.
# 
# COMUNALIDAD (Loadings-Saturaciones): Porcentaje de la variabilidad de la variable explicada por ese Factor.
# 
# FACTOR1 - FACTOR2: Algebraicamente, un factor se estima mediante una combinaci?n lineal de variables observadas. Cuando se encuentran los factores de "mejor ajuste", 
# debe recordarse que estos factores no son ?nicos. Se puede demostrar que cualquier rotaci?n de los factores que mejor se ajuste es tambi?n el mejor factor. 
# La rotaci?n de factores se utiliza para ajustar la varianza que explicar? el Factor.
# 
# Si todos los factores explican conjuntamente un gran porcentaje de varianza en una variable dada, esa variable tiene una alta comunalidad (y por lo tanto una singularidad baja)
# 
# SS loadings: La saturaci?n acumulada
# 
# Proportion Var: Proporci?n de la varianza

# Cumulative Var: Varianza acumulada

#puntuaciones o pesos que son considerados como las saturaciones por cada observaci?n de las variables ingresadas al c?lculo del AF:

factanal(estudiantes, factors = 2, rotation = "none", scores = "regression")$scores

#library(nFactors)
#fa.parallel(R,n.obs=20,fa="fa",fm="mle")

# ev <- eigen(R)
# ap <- parallel(subject=nrow(estudiantes),var=ncol(estudiantes), rep=100,cent=.05)
# nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
# plotnScree(nS)

# factores_best<-target.rot(modelo)
# factores_best
# plot(factores_best)
# modelo$fit#propoci?n de varianza explicada por los factores
