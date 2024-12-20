# Carga de datos
path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica I/Prueba/health.csv"
health<- read.csv(path, header = T,sep = ";")

# Vemos el número de datos observaciones que tenemos
# Disponemos de 7 variables y 508 observaciones
str(health)

##########################
####### Apartado 5 ####### 
##########################

# Para la variable log_virus --> 30.5% de valores missing (155 obs)
# Para la variable damage --> 7.68% de valores missing (39 obs)
# Para la variable healthy --> 4.13% de valores missing (21 obs)
# Para la variable income --> 2.76% de valores missing (14 obs)
miss_var_summary(health)

# Grafico
# Se aprecia el gráficamente el volumen de datos faltantes en cada 
# una de las variables. Cuanto más región negra, más valores faltantes.
vis_miss(health, cluster = TRUE)

##########################
####### Apartado 6 ####### 
##########################

# Grafico
gg_miss_upset(health)

# Cada barra representa una combinación única de variables con valores faltantes.
# Por ejemplo, la variable logvirus tiene un total de 116 valores únicos nulos
# o la variable damage con 28.
# En cambio, logvirus, tiene compartidos valores nulos con otras variables, por ejemplo, la 
# variabe de healthy, damage e income (es decir, cuando en logvirus es nulo también esas 3 variables)
# Incluso hay casos en los que cuando logvirus es nulo, también en damage e income.
# Esto también ocurre con la variable de damage e income, hay ciertos valores nulos que "comparten",
# hay ciertos valores nulos que ocurren simultaneamente en damage e income.

##########################
####### Apartado 7 ####### 
##########################

# En mi opinión, creo que no. Por ejemplo, para la variable de logvirus, parece que
# hay una cierta relación de que, cuando esta es nula, también lo es en otras como income,
# healthy o damage. 
# Por otro lado, también esta el caso de damage e income, parece que cuando alguna de las 
# dos es nula, hay más posibildades de encontrar un valor faltante en la otra.

##########################
####### Apartado 8 ####### 
##########################

library(dplyr)

# Regresión estocastica
health_imputed_slm <-
  health %>% 
  bind_shadow(only_miss = TRUE) %>% # agrega las variables con _NA
  add_label_shadow() %>% # agrega una etiqueta en una columna nueva con "Missing" o "Not Missing"
  impute_lm(healthy ~ age, add_residual = "normal") # imputar valores de healthy usando age 

# Dispersión
ggplot(health_imputed_slm, 
         aes(x = healthy,
             y = age,
             color = any_missing)) +
  geom_point()

# Cajas para healthy
ggplot(health_imputed_slm, aes(y = healthy)) +
  geom_boxplot() +
  theme_minimal() 

# La aproximación parece bastante buena pues los datos que aparecen missing tienen
# valores muy semejantes a los conocidos. La predicción de la regresión estocastica parece
# que es capaz de ajustarse al valor real de los datos que nos faltan en el dataset.


