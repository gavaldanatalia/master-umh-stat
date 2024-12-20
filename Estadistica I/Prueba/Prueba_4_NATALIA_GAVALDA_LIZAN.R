# Carga de datos
path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Prueba/fisico.csv"
fisico<- read.csv(path, header = T,sep = ";")

str(fisico)
summary(fisico)

##########################
####### Apartado 17 ######
##########################

# Seleccionamos las variables numéricas
fisico <- fisico[, 2:16]

# Análisis gráfico
describe(fisico)

# Vemos las correlaciones
R <- cor(fisico)
corrplot(R,method = "square")

# Test de esfericidad de Bartlett e indica si es conveniente realizar un Análisis Factorial
# Como el pvalor < 0.05, sí que es adecuado aplicar AF
cortest.bartlett(R, n = 20)

# Probamos con KMO también
# Resulta de 0.64, aunque el valor no es muy alto, podemos aplicar AF.
KMO(R)

# Por lo tanto, la conclusión es que sí que aplciamos AF

##########################
####### Apartado 18 ######
##########################

library(nFactors)
library(parameters)

result_nfactors <- n_factors(fisico, type = "FA")
plot(result_nfactors)

# Parece que podemos elegir 1 o 2 factores, debido a que están muy igualadas las barras
# Procedemos a utilizar 2 factores.

scree(R) # Aquí se aprecia que pueden ser 2 también, es cuando los valores propios < 1

# Conclusión: Nos quedamos con 2 factores.

##########################
####### Apartado 19 ######
##########################

# Modelo simple sin rotación
modelo1<-fa(fisico,
            nfactors = 3,
            rotate = "none",
            fm="pa",
            max.iter = 200)


# comunalidades - % varianza explicada por variable
modelo1$communality 
# Parece que con este primer modelo todas las variables se explican bien salvo:
# est_visual, cap_resp y resistencia. Tienen un % de varianza explicada muy bajo.

# Si hacemos el corte, vemos que la FA3 no le contribuye ninguna de las variables
# Lo que se espera es que todas las PA sean explicadas por alguna de las variables originales
print(modelo1$loadings,cut=0.4)
fa.diagram(modelo1)

# Probamos con otro tipo de rotación, por ejemplo:
# Rotación ortogonal
modelo2<-fa(fisico,
            nfactors = 3,
            rotate = "varimax",
            fm="pa",
            max.iter = 200)
modelo2$communality 
# Parece que con este primer modelo todas las variables se explican bien salvo:
# est_visual, presion_art, cap_resp y resistencia. Tienen un % de varianza explicada muy bajo.

# Si hacemos el corte, nos queda que la variable est_visual, resistencia y presion_art no contribuyen
# a ninguno de los factores. En cambio mejoramos en que la componente 3 es capaz de explicar
# más varianza y el modelo se encunetra más repartido entre las PA
print(modelo2$loadings,cut=0.4)
fa.diagram(modelo2)

# Conclusión: Nos quedamos con el modelo de VARIMAX

##########################
####### Apartado 20 ######
##########################

# Calculamos las puntuacioes con el método de regression
factanal_scores = factanal(fisico, factors = 3, rotation = "varimax", scores = "regression")$scores

# Hacemos el gráfico de los sujetos con el Factor 1 y el Factor 2
plot(
  factanal_scores[, 1],  # Puntuaciones del Factor 1
  factanal_scores[, 2],  # Puntuaciones del Factor 2
  xlab = "Factor 1",
  ylab = "Factor 2",
  main = "Representación de los sujetos en los ejes factoriales",
  pch = 19, col = "blue"
)
