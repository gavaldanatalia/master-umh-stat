#####EJERCICIO VALORES FALTANTES
library(naniar)
library(dplyr)
library(ggplot2)
library(simputation)

### APARTADO 1.
# Haz un resumen de los datos faltantes de cada variable y de cada caso. 
# ¿Qué variable presenta más valores faltantes? ¿Qué podría implicar esto sobre 
# la calidad de los datos o la recolección de los mismos?

str(airquality)

miss_var_summary(airquality) # cada fila representa una variable con el número y el porcentaje de NA's

miss_case_summary(airquality) # cada fila representa una fila con el número y el porcentaje de NA's

miss_var_table(airquality) # devuelve un dataframe con el número de NA's en una variable: número y porcentaje de variables afectadas

miss_case_table(airquality) # devuelve un dataframe con el número de NA's en una variable: número y porcentaje de filas afectadas

####Apartado 2
# 2. Visualiza los valores faltantes en los diferentes gráficos trabajados en clase. 
# Comenta los resultados. ¿Notas algún patrón en los valores faltantes entre las 
# variables? ¿Parece que los valores faltantes están correlacionados entre sí?

vis_miss(airquality)

vis_miss(airquality, cluster = TRUE)

gg_miss_var(airquality) 

gg_miss_case(airquality)

gg_miss_upset(airquality)


### APARTADO 3. 
# 3.	Visualiza los valores faltantes facetada por la variable "Month". 
# ¿En qué meses se concentran los valores faltantes? ¿Podría haber alguna razón
# estacional para esta distribución?
gg_miss_var(airquality, facet = Month) # visualización faceteada por la variable "Month"
gg_miss_fct(x = airquality, fct = Month) # explorar cómo cambian los valores faltantes en cada variable en un factor


### APARTADO 4
# Genera la matriz sombra de la base de datos

as_shadow(airquality)

# crear datos nabulares uniendo la matriz sombra a los datos con `bind_shadow()`.
bind_shadow(airquality)



### APARTADO 5
# 5.	Calcula la media de la variable Wind diferenciada por los datos completos 
# y datos faltantes de la variable Ozone. ¿Existe una diferencia notable entre 
# ambas medias? Si existe, ¿qué podría estar indicando esta diferencia sobre la 
# relación entre Wind y Ozone?

airquality %>% 
  bind_shadow() %>% 
  group_by(Ozone_NA) %>% 
  summarize(mean = mean(Wind))

airquality %>% 
  bind_shadow() %>% 
  group_by(Ozone_NA) %>% 
  summarize(mean = mean(Temp))

### APARTADO 6
# Visualiza la distribución de la variable Wind en función de los valores faltantes 
# de la variable Ozone mediante un gráfico de cajas. Visualiza también la distribución 
# de la variable Temp en los mismos términos. ¿Qué diferencias encuentras entre las 
# distribuciones de Wind y Temp cuando Ozone es faltante o completo? ¿Qué hipótesis 
# podrías plantear sobre las posibles causas de los valores faltantes?

airquality %>% 
  bind_shadow() %>% 
  ggplot(aes(y = Wind,
             color = Ozone_NA)) +
  geom_boxplot()  # visualizar variable Wind cuando hay datos para Ozone y para cuando no los hay

airquality %>% 
  bind_shadow() %>% 
  ggplot(aes(y = Temp,
             color = Ozone_NA)) +
  geom_boxplot()  # visualizar variable Temp cuando hay datos para Ozone y para cuando no los hay

### APARTADO 7
# Realiza un gráfico de dispersión de la variable Wind en función de Temp, 
# en el gráfico deben estar diferenciados los registros que sean faltantes o 
# completos en la variable Ozone. ¿Hay patrones claros en la relación Wind-Temp 
# que puedan estar relacionados con los valores faltantes de Ozone? ¿Qué podría 
# significar esto?

airquality %>% 
  bind_shadow() %>% 
  ggplot(aes(x = Temp,
             y = Wind,
             color = Ozone_NA)) +
  geom_point() # visualizar variables Temp y Wind cuando hay datos para Ozone y para cuando no los hay

### APARTADO 8
# Realiza el test de Little y concluye si los datos faltantes son MCAR. 
# Si el test indica que los datos no son MCAR, ¿qué estrategias considerarías para 
# tratar los valores faltantes? ¿Cómo cambiaría tu enfoque dependiendo del resultado?

mcar_test(airquality)

#### APARTADO 9
# Genera una base de datos a partir de la original que se hayan eliminado los 
# datos faltantes. ¿Cuántas observaciones se pierden al eliminar las filas con 
# valores faltantes? ¿Cómo afecta esto al tamaño de la muestra y la 
# representatividad del análisis?

airquality_cc <- airquality %>% 
  na.omit

str(airquality_cc)
str(airquality)

#### APARTADO 10
# Imputa los valores faltantes por la media, en aquellas variables que 
# haya valores faltantes. ¿Qué impacto tiene esta imputación sobre las distribuciones 
# de las variables imputadas? 

airquality_imputed_mean <-
  airquality %>% 
  bind_shadow(only_miss = TRUE) %>% 
  impute_mean_all() %>% 
  add_label_shadow()

head(airquality_imputed_mean)  
View(airquality_imputed_mean)


### APARTADO 11
# 11.	Evalúa las imputaciones realizadas en el apartado anterior mediante los 
# gráficos pertinentes. ¿Las distribuciones de las variables imputadas son 
# similares a las originales? ¿Qué conclusiones puedes extraer sobre la validez 
# de la imputación por la media?

ggplot(airquality_imputed_mean, 
       aes(x = Ozone_NA,
           y = Ozone)) +
  geom_boxplot()

# esta visualización nos muestra que no hay variación en la dispersión de los puntos, 
# los valores imputados están dentro de un rango sensato de los datos.
ggplot(airquality_imputed_mean, 
       aes(x = Ozone,
           y = Solar.R,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_mean, 
       aes(x = Wind,
           y = Ozone,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_mean, 
       aes(x = Temp,
           y = Ozone,
           color = any_missing)) +
  geom_point()

#Variable Ozone
ggplot() +
  geom_density(data = airquality, aes(x = Ozone, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_mean, aes(x = Ozone, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Ozone antes y después de imputación por la media",
       x = "Ozone", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))

# Variable Solar.R
ggplot() +
  geom_density(data = airquality, aes(x = Solar.R, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_mean, aes(x = Solar.R, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Solar.R antes y después de imputación por la media",
       x = "Solar.R", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))


#### APARTADO 12
# 12.	Imputa los valores faltantes en las variables Ozone y Solar.R,
# por regresión lineal y por regresión estocástica. Utiliza como variables 
# independientes Wind y Temp. Evalúa las imputaciones realizadas mediante los 
#gráficos pertinentes. ¿Las imputaciones realizadas por regresión preservan las 
# relaciones entre variables? ¿Cuál de los métodos (lineal o estocástico) parece 
#más adecuado según los gráficos obtenidos?

str(airquality)
airquality2 <- airquality %>% 
  mutate(Ozone = as.numeric(Ozone),
         Solar.R =as.numeric(Solar.R),
         Temp =as.numeric(Temp))

#regresión lineal
airquality_imputed_lm <-
  airquality2 %>% 
  bind_shadow(only_miss = TRUE) %>% # agrega las variables con _NA
  add_label_shadow() %>% # agrega una etiqueta en una columna nueva con "Missing" o "Not Missing"
  impute_lm(Ozone ~ Wind + Temp) %>% # imputar valores de Ozone usando las variables Wind + Temp 
  impute_lm(Solar.R ~ Wind + Temp) # imputar valores de Solar.R usando las variables Wind + Temp 


ggplot(airquality_imputed_lm, 
       aes(x = Wind,
           y = Ozone,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_lm, 
       aes(x = Temp,
           y = Ozone,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_lm, 
       aes(x = Wind,
           y = Solar.R,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_lm, 
       aes(x = Temp,
           y = Solar.R,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_lm,
       aes(x = Solar.R,
           y = Ozone,
           color = any_missing)) + # colorea los valores imputados
  geom_point()

ggplot() +
  geom_density(data = airquality, aes(x = Ozone, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_lm, aes(x = Ozone, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Ozone antes y después de imputación por la media",
       x = "Ozone", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))

ggplot() +
  geom_density(data = airquality, aes(x = Solar.R, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_lm, aes(x = Solar.R, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Solar.R antes y después de imputación por la media",
       x = "Solar.R", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))


#regresión estocástica
airquality_imputed_slm <-
  airquality2 %>% 
  bind_shadow(only_miss = TRUE) %>% # agrega las variables con _NA
  add_label_shadow() %>% # agrega una etiqueta en una columna nueva con "Missing" o "Not Missing"
  impute_lm(Ozone ~ Wind + Temp, add_residual = "normal") %>% # imputar valores de Ozone usando las variables Wind + Temp 
  impute_lm(Solar.R ~ Wind + Temp, add_residual = "normal") # imputar valores de Solar.R usando las variables Wind + Temp 



ggplot(airquality_imputed_slm, 
       aes(x = Wind,
           y = Ozone,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_slm, 
       aes(x = Temp,
           y = Ozone,
           color = any_missing)) +
  geom_point()

ggplot(airquality_imputed_slm, 
       aes(x = Wind,
           y = Solar.R,
           color = any_missing)) +
  geom_point()


ggplot(airquality_imputed_slm, 
       aes(x = Temp,
           y = Solar.R,
           color = any_missing)) +
  geom_point()

ggplot() +
  geom_density(data = airquality, aes(x = Ozone, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_slm, aes(x = Ozone, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Ozone antes y después de imputación por la media",
       x = "Ozone", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))

ggplot() +
  geom_density(data = airquality, aes(x = Solar.R, color = "Original"), na.rm = TRUE) +
  geom_density(data = airquality_imputed_slm, aes(x = Solar.R, color = "Imputado"), na.rm = TRUE) +
  labs(title = "Distribución de Solar.R antes y después de imputación por la media",
       x = "Solar.R", y = "Densidad") +
  scale_color_manual(name = "Tipo", values = c("Original" = "blue", "Imputado" = "red"))



