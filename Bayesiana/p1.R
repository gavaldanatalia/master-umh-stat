#remotes::install_github("michael-franke/aida-package")
library(brms)
library(aida)
library(tidyverse)
library(dplyr)

################################
######### Modelo 1 #############
################################
fit_temperature <- brm(
  formula = avg_temp ~ year, # relación lineal entre las variables
  data = aida::data_WorldTemp, # data
  )

# resumen del modelo
summary(fit_temperature)

# Obtenemos las distribuciones posteriores
post_samples_temperature <- as_draws_df(fit_temperature) %>%
  dplyr::select(-lp__, -lprior)

head(post_samples_temperature)
dim(post_samples_temperature)

# Estadisticas de las distribuciones
summary_table <- map_dfr(post_samples_temperature,
                         aida::summarize_sample_vector) %>%
  mutate(Parameter = colnames(post_samples_temperature))
summary_table

# Gráficos
# Distribuciones a posteriori de b_year, b_intercept y sigma
post_samples_temperature[, 1:3] %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x = value)) +
  geom_density() +
  facet_wrap(~name, scales = "free")

# Obtener resumen de medias por b_year, por b_intercept y por sigma
post_samples_temperature %>%
  summarise_all(mean)

# Modelo en Stan
modelo<-brms::stancode(fit_temperature)
substr(modelo, start=1, stop=nchar(modelo))

# La función brms::prior_summary muestra qué priors ha asumido
# (implícitamente) un modelo ajustado con brms.
# Esta salida nos indica que brms usó una distribución Student’s t
# para el intercepto y la desviación estándar
# También muestra que todos los coeficientes de pendiente
# (abreviados aquí como “b”) tienen un prior plano (no informativo).
brms::prior_summary(fit_temperature)

################################
######### Modelo 2 #############
################################

# ¿Por qué hacemos este modelo 2?
# Si queremos cambiar la prior para cualquier parámetro o grupo de
# parámetros, podemos usar el argumento prior en la función brm
# junto con la función prior().
# La sintaxis para las distribuciones dentro de prior() sigue la de
# Stan, según la referencia de funciones de Stan.
# El ejemplo siguiente establece la prior para el coeficiente de la
# pendiente a una distribución Student’s t muy estrecha con media
# -0.01 y desviación estándar 0.001.


# Modelo 2 con prior escéptico: asume que una pendiente negativa es más probable
# "El mundo se ha enfriado a lo largo de los años"
fit_temperature_skeptical <- brm(
  # specify what to explain in terms of what using the formula syntax
  formula = avg_temp ~ year,
  # which data to use
  data = aida::data_WorldTemp,
  # hand-craft priors for slope
  prior = prior(student_t(1, -0.01, 0.001), coef = year)
)

# Obtenemos las distribuciones posteriores
post_samples_temperature_skeptical <-
   as_draws_df(fit_temperature_skeptical) %>%
         select(-lp__, -lprior)
post_samples_temperature_skeptical
 
# Estadisticas de las distribuciones
summary_table <- map_dfr(post_samples_temperature_skeptical,
                          aida::summarize_sample_vector) %>%
     mutate(Parameter = colnames(post_samples_temperature_skeptical))
summary_table
