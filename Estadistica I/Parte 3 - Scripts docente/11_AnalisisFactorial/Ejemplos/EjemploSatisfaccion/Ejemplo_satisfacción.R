library(psych)
library(dplyr)
library(corrplot)
library(GPArotation)

satisfac <- read.table("satisfaccion.csv", header = T, sep = ";")

str(satisfac)

datos <- satisfac[, 6:12]

str(datos)

#1. Visualiza la estructura de la base de datos y realiza un resumen descriptivo de las variables.
describe(datos)
R <- cor(datos); R

corrplot(R,method = "square")

#2. Realiza el Test de esfericidad de Bartlett e indica si es conveniente realizar un Análisis Factorial.
###KMO Y BARLETT TEST

KMO(R)
cortest.bartlett(R, n = 100 )


#3. Elige el número de factores que creas que explican mejor los datos.
scree(R)
eigen(R);cumsum(eigen(R)$values)/sum(diag(R))

library(nFactors)
library(parameters)

result_nfactors <- n_factors(datos, type = "FA")
plot(result_nfactors)
result_nfactors
as.data.frame(result_nfactors)
summary(result_nfactors)

#4. Realiza el Análisis Factorial con el número de factores que has elegido
#Estimacion por factores principales
modelo1<-fa(datos,
            nfactors = 3,
            rotate = "none",
            fm="pa") # 

modelo1$communality                 #comunalidades
modelo1$uniquenesses               #unicidades
modelo1$loadings

print(modelo1$loadings,cut=0.4) 

residuals(modelo1)

fa.diagram(modelo1)

#Las saturaciones estan un poco mezcladas, realizamos una rotacion ortogonal

modelo1_varimax<-fa(datos,nfactors = 3,
                    rotate = "varimax",
                    fm="pa")

modelo1_varimax$communality  #vemos que no varian

#modelo1_varimax
print(modelo1_varimax$loadings,cut=0.4)

fa.diagram(modelo1_varimax)

plot(modelo1_varimax$loadings[,1:2])

text(modelo1_varimax$loadings[,1],modelo1_varimax$loadings[,2],
     labels = row.names(modelo1_varimax$loadings),
     cex = 0.6, pos = 1, col = "red")

plot(modelo1_varimax$loadings[,c(1,3)])

text(modelo1_varimax$loadings[,1],modelo1_varimax$loadings[,3],
     labels = row.names(modelo1_varimax$loadings),
     cex = 0.6, pos = 1, col = "red")

fa.plot(modelo1_varimax)

# legend(x ="topleft", inset = 0.15,cex = 0.5,legend = c("P1-P2", "P3-P4-P5", "P6-P7"), fill= c(1,2,4))

#5.Calcula las puntuaciones de los sujetos mediante el modelo factorial estimado 

modelo1_varimax$scores


#6.Representa a los pacientes en ejes formados por los factores encontrados.
library(scatterplot3d)

scatterplot3d(modelo1_varimax$scores[,1:3],
              pch = 16,
              type="h",
              box=FALSE)

biplot(modelo1_varimax)

#7. Introduce la variable P8 en los scores. Interpreta los resultados obtenidos.

satisfac$S <- as.factor(satisfac$S)
satisfac$C <- as.factor(satisfac$C)
satisfac$D <- as.factor(satisfac$D)
satisfac$P8 <- as.factor(satisfac$P8)

str(satisfac)
head(satisfac)

colors <- c("purple", "orange")
colors <- colors[as.numeric(satisfac$P8)]
scatterplot3d(modelo1_varimax$scores[,1:3], 
              color = colors, 
              pch = 16,
              type="h",
              box=FALSE)
legend("bottomright", legend = levels(satisfac$P8),
       col =  c("purple", "orange"), pch = 16,cex = 0.6)


#8. Introduce la variable Cirugía en los scores. Interpreta los resultados obtenidos.

colors <- c("purple", "orange")
colors <- colors[as.numeric(satisfac$C)]
scatterplot3d(modelo1_varimax$scores[,1:3], 
              color = colors, 
              pch = 16,
              type="h",
              box=FALSE)
legend("bottomright", legend = levels(satisfac$C),
       col =  c("purple", "orange"), pch = 16,cex = 0.6)

#9.Introduce la variable Sexo en los scores. Interpreta los resultados obtenidos.

colors <- c("purple", "orange")
colors <- colors[as.numeric(satisfac$S)]
scatterplot3d(modelo1_varimax$scores[,1:3], 
              color = colors, 
              pch = 16,
              type="h",
              angle = 55,
              box=FALSE)
legend("bottomright", legend = levels(satisfac$S),
       col =  c("purple", "orange"), pch = 16,cex = 0.6)


#10.Introduce la variable Departamento en los scores. Interpreta los resultados obtenidos.

colors <- c("purple", "orange", "grey", "green")
colors <- colors[as.numeric(satisfac$D)]
scatterplot3d(modelo1_varimax$scores[,1:3], 
              color = colors, 
              pch = 16,
              type="h",
              box=FALSE)

legend("bottomright", legend = levels(satisfac$D),
       col =  c("purple", "orange", "grey", "green"), pch = 16,cex = 0.6)




# plot(modelo1_varimax$scores[,1:2])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,2],
#      labels = satisfac$C ,
#      cex = 0.6, pos = 1)
# 
# plot(modelo1_varimax$scores[,c(1,3)])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,3],
#      labels = satisfac$C ,
#      cex = 0.6, pos = 1, col = c("red","blue"))


# plot(modelo1_varimax$scores[,1:2])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,2],
#      labels = satisfac$P8 ,
#      cex = 0.6, pos = 1)
# 
# plot(modelo1_varimax$scores[,c(1,3)])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,3],
#      labels = satisfac$P8 ,
#      cex = 0.6, pos = 1, col = c("red","blue"))


# plot(modelo1_varimax$scores[,1:2])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,2],
#      labels = satisfac$S ,
#      cex = 0.6, pos = 1, col = c("red","blue"))
# 
# plot(modelo1_varimax$scores[,c(1,3)])
# 
# text(modelo1_varimax$scores[,1],modelo1_varimax$scores[,3],
#      labels = satisfac$S ,
#      cex = 0.6, pos = 1, col = c("red","blue"))
