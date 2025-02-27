## ----setup, include=FALSE----
knitr::opts_chunk$set(echo = TRUE, 
                      warning = F,
                      message = F,
                      cache=T, 
                      tidy.opts=list(width.cutoff=60),
                      tidy=FALSE)
library(xtable)
library(knitr)
library(bayesrules)
options(width=60)
library(bayesrules) # paquete del Libro Bayes rules
library(tidyverse) 
#library(rstan) # cargando el stan
#library(rstanarm) # cargando el stan
library(bayesplot) # para dibujar resultados
library(tidybayes) # facilitar el uso de tydy en Bayesiano
library(janitor) # limpieza de datos
library(broom.mixed) # adapta modelos al formato "tidy"
library(ggpubr) # para utilizar ggarrange


## ----eval=FALSE-------------
#  #remotes::install_github("michael-franke/aida-package")
# library(brms)
# library(aida)
# fit_temperature <- brm(
#   # especificar qué variable explicar en términos de cuál,
#   # usando la sintaxis de fórmulas
#   formula = avg_temp ~ year,
#   # qué datos usar
#   data = aida::data_WorldTemp)


## ----eval=FALSE-------------
# library(brms)
# library(aida)
# fit_temperature <- brm(
#   # Especificar qué variable explicar en términos de cuál,
#   # usando la sintaxis de fórmulas
#   formula = avg_temp ~ year,
#   # Qué datos usar
#   data = aida::data_WorldTemp)


## ----echo=FALSE, message=F, warning=F----
library(brms)
library(aida)
output <- capture.output(fit_temperature <- brm(
  formula = avg_temp ~ year,
  data = aida::data_WorldTemp,
  chains = 4,
  iter = 5000 * 2,
  seed = 84735,
  refresh = 0
))


## ----eval=FALSE-------------
# fit_temperature


## ----echo=FALSE-------------
# Ejemplo de salida (comentario: esta salida es ilustrativa)
fit_temperature


## ----eval=FALSE, tidy=F-----
# post_samples_temperature <-
#   as_draws_df(fit_temperature) %>%
#   dplyr::select(-lp__, -lprior)
# head(post_samples_temperature)


## ----echo=FALSE-------------
post_samples_temperature <- as_draws_df(fit_temperature) %>%
  dplyr::select(-lp__, -lprior)
head(post_samples_temperature)


## ----eval=FALSE, tidy=F-----
# map_dfr(post_samples_temperature[, 1:3],
#         aida::summarize_sample_vector) %>%
#           mutate(Parameter =
#                    colnames(post_samples_temperature[, 1:3]))


## ----echo=FALSE-------------
map_dfr(post_samples_temperature[, 1:3], aida::summarize_sample_vector) %>%
  mutate(Parameter = colnames(post_samples_temperature[, 1:3]))


## ----eval=FALSE-------------
# post_samples_temperature %>%
#   pivot_longer(cols = everything()) %>%
#   ggplot(aes(x = value)) +
#   geom_density() +
#   facet_wrap(~name, scales = "free")


## ----echo=FALSE, out.height="80%"----
post_samples_temperature[, 1:3] %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x = value)) +
  geom_density() +
  facet_wrap(~name, scales = "free")


## ---------------------------
modelo<-brms::stancode(fit_temperature)


## ----echo=F-----------------
substr(modelo, start=1, stop=630)



## ----echo=F-----------------
substr(modelo, start=631, stop=1035)


## ----echo=F-----------------
substr(modelo, start=1035, stop=nchar(modelo))


## ----eval=FALSE-------------
# brms::prior_summary(fit_temperature)


## ----echo=FALSE-------------
brms::prior_summary(fit_temperature)


## ----eval=T, resutls="hide"----
output2 <- capture.output(fit_temperature_skeptical <- brm(
  # specify what to explain in terms of what using the formula syntax
  formula = avg_temp ~ year,
  # which data to use
  data = aida::data_WorldTemp,
  # hand-craft priors for slope
  prior = prior(student_t(1, -0.01, 0.001), coef = year)
))


## ----eval=FALSE, tidy=F-----
# map_dfr(post_samples_temperature[, 1:3],
#         aida::summarize_sample_vector) %>%
#   mutate(Parameter =
#            colnames(post_samples_temperature[, 1:3]))


## ----echo=FALSE, tidy=F-----
map_dfr(post_samples_temperature[, 1:3], aida::summarize_sample_vector) %>%
  mutate(Parameter = colnames(post_samples_temperature[, 1:3]))


## ----eval=FALSE, tidy=F-----
# post_samples_temperature_skeptical <-
#   as_draws_df(fit_temperature_skeptical) %>%
#         select(-lp__, -lprior)
# 
# summary_table <- map_dfr(post_samples_temperature_skeptical,
#                          aida::summarize_sample_vector) %>%
#     mutate(Parameter = colnames(post_samples_temperature_skeptical))
# 


## ----echo=F, tidy=F---------
post_samples_temperature_skeptical <-
  as_draws_df(fit_temperature_skeptical) %>%
        select(-lp__, -lprior)

summary_table <- map_dfr(post_samples_temperature_skeptical,
                         aida::summarize_sample_vector) %>%
    mutate(Parameter = colnames(post_samples_temperature_skeptical))

kable(summary_table[1:3,], col.names = c("Parameter","P2.5%","Mean","P97.5%"), digits=4)


## ----eval=FALSE, tidy=F-----
# output3 <- capture.output(fit_temperature_ridiculous <- brm(
#   # specify what to explain in terms of what using the formula syntax
#   formula = avg_temp ~ year,
#   # which data to use
#   data = aida::data_WorldTemp,
#   # hand-craft a very strong prior for slope
#   prior = prior(normal(5, 0.01), coef = year)
# ))
# # Usar as_draws_df() (alternativa recomendada a posterior_samples)
# post_samples_temperature_ridiculous <- as_draws_df(fit_temperature_ridiculous) %>%
#   select(-lp__, -lprior)
# 
# # Crear el resumen y asignar los nombres correctos (suponiendo que el resumen tiene 4 filas)
# summary_table <- map_dfr(post_samples_temperature_ridiculous, aida::summarize_sample_vector) %>%
#   mutate(Parameter = colnames(post_samples_temperature_ridiculous))
# 
# 
# # Ver el resumen
# kable(summary_table[1:3,], col.names = c("Parameter","P2.5%","Mean","P97.5%"), digits=4)


## ----echo=FALSE, tidy=F-----
output3 <- capture.output(fit_temperature_ridiculous <- brm(
  # specify what to explain in terms of what using the formula syntax
  formula = avg_temp ~ year,
  # which data to use
  data = aida::data_WorldTemp,
  # hand-craft a very strong prior for slope
  prior = prior(normal(5, 0.01), coef = year)
))
# Usar as_draws_df() (alternativa recomendada a posterior_samples)
post_samples_temperature_ridiculous <- as_draws_df(fit_temperature_ridiculous) %>%
  select(-lp__, -lprior)

# Crear el resumen y asignar los nombres correctos (suponiendo que el resumen tiene 4 filas)
summary_table <- map_dfr(post_samples_temperature_ridiculous, aida::summarize_sample_vector) %>%
  mutate(Parameter = colnames(post_samples_temperature_ridiculous))


# Ver el resumen
kable(summary_table[1:3,], col.names = c("Parameter","P2.5%","Mean","P97.5%"), digits=4)


## ----eval=F-----------------
# library(bayesplot)
# mcmc_hist(post_samples_temperature_ridiculous,
#           pars = c("b_Intercept", "b_year", "sigma"))


## ----eval=F, out.height="80%"----
# library(bayesplot)
# mcmc_hist(post_samples_temperature_ridiculous,
#           pars = c("b_Intercept", "b_year", "sigma"))


## ----eval=T-----------------
samples_post_pred_temperature <- brms::posterior_predict(fit_temperature)
dim(samples_post_pred_temperature)


## ----eval=FALSE-------------
# # Crear un tibble con nuevos valores de predictores
# X_new <- tribble(
#   ~year,
#   2025,
#   2040
# )
# 
# # Obtener predicciones muestrales del modelo bayesiano
# post_pred_new <- brms::posterior_predict(fit_temperature, X_new)
# 
# # Obtener un resumen (bayesiano) de estas muestras posteriores
# rbind(
#   aida::summarize_sample_vector(post_pred_new[,1], "2025"),
#   aida::summarize_sample_vector(post_pred_new[,2], "2040")
# )


## ----echo=FALSE-------------
# Crear un tibble con nuevos valores de predictores
X_new <- tribble(
  ~year,
  2025,
  2040
)

# Obtener predicciones muestrales del modelo bayesiano
post_pred_new <- brms::posterior_predict(fit_temperature, X_new)

# Obtener un resumen (bayesiano) de estas muestras posteriores
rbind(
  aida::summarize_sample_vector(post_pred_new[,1], "2025"),
  aida::summarize_sample_vector(post_pred_new[,2], "2040")
)


## ----eval=F-----------------
# library(tidyverse)
# 
# # Supongamos que los datos originales son:
# data_original <- aida::data_WorldTemp
# 
# # Obtener resumen de la predicción (media y RC al 95%)
# pred_summary <- as_tibble(post_pred_new) %>%  #pred_summary  # post_pred_new
#   pivot_longer(cols = everything(), names_to = "id", values_to = "pred") %>%
#   group_by(id) %>%
#   summarise(
#     avg_temp = mean(pred),
#     lower = quantile(pred, 0.025),
#     upper = quantile(pred, 0.975)
#   ) %>%
#   mutate(year = X_new$year)
# 
# # Gráfico con ggplot2
# ggplot(data_original, aes(x = year, y = avg_temp)) +
#   geom_point(color = "black", alpha = 0.7) +  # Datos originales
#   geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dashed") +  # Tendencia original
#   geom_line(data = pred_summary, aes(x = year, y = avg_temp), color = "red", linewidth = 1) +  # Predicción media
#   geom_ribbon(data = pred_summary, aes(x = year, ymin = lower, ymax = upper),
#               fill = "red", alpha = 0.2) +  # Región de credibilidad
#   labs(
#     title = "Predicción Bayesiana con Región de Credibilidad",
#     x = "Año",
#     y = "Temperatura Media"
#   ) +
#   theme_minimal()


## ----echo=F-----------------
library(tidyverse)

# Supongamos que los datos originales son:
data_original <- aida::data_WorldTemp

# Obtener resumen de la predicción (media y RC al 95%)
pred_summary <- as_tibble(post_pred_new) %>%  #pred_summary  # post_pred_new
  pivot_longer(cols = everything(), names_to = "id", values_to = "pred") %>%
  group_by(id) %>%
  summarise(
    avg_temp = mean(pred),
    lower = quantile(pred, 0.025),
    upper = quantile(pred, 0.975)
  ) %>%
  mutate(year = X_new$year)

# Gráfico con ggplot2
ggplot(data_original, aes(x = year, y = avg_temp)) +
  geom_point(color = "black", alpha = 0.7) +  # Datos originales
  geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dashed") +  # Tendencia original
  geom_line(data = pred_summary, aes(x = year, y = avg_temp), color = "red", linewidth = 1) +  # Predicción media
  geom_ribbon(data = pred_summary, aes(x = year, ymin = lower, ymax = upper), 
              fill = "red", alpha = 0.2) +  # Región de credibilidad
  labs(
    title = "Predicción Bayesiana con Región de Credibilidad",
    x = "Año",
    y = "Temperatura Media"
  ) +
  theme_minimal()


## ----eval=T-----------------
output <- capture.output(fit_temperature_weakinfo <- brm(
  # Especificar qué variable explicar en términos de cuál (sintaxis de fórmula)
  formula = avg_temp ~ year,
  # Datos a utilizar
  data = aida::data_WorldTemp,
  # Prior poco informativo para la pendiente
  prior = prior(student_t(1, 0, 1), coef = year),
  # Obtener muestras de la prior
  sample_prior = "yes",
  # Aumentar el número de iteraciones para mayor precisión
  iter = 20000
))


## ----eval=F-----------------
# # Obtener las muestras posteriores usando as_draws_df
# posterior_draws <- as_draws_df(fit_temperature_weakinfo)
# 
# # Extraer la columna correspondiente al parámetro 'b_year' y resumirla
# posterior_draws %>%
#   pull(b_year) %>%
#   aida::summarize_sample_vector()


## ----echo=F-----------------
# Obtener las muestras posteriores usando as_draws_df
posterior_draws <- as_draws_df(fit_temperature_weakinfo)

# Extraer la columna correspondiente al parámetro 'b_year' y resumirla
posterior_draws %>%
  pull(b_year) %>%
  aida::summarize_sample_vector()


## ----eval=FALSE-------------
# hypothesis(fit_temperature_weakinfo, "year > 0")


## ----echo=FALSE-------------
hypothesis(fit_temperature_weakinfo, "year > 0")


## ----eval=FALSE-------------
# hypothesis(fit_temperature_weakinfo,
#            "year = 0.005")


## ----echo=FALSE-------------
hypothesis(fit_temperature_weakinfo, "year = 0.005")


## ----eval=FALSE-------------
# # Extraer únicamente las columnas relevantes del conjunto de datos
# data_ST_excerpt <- aida::data_ST %>%
#   filter(correctness == "correct") %>%
#   select(RT, condition)
# 
# # Mostrar las primeras 5 líneas
# head(data_ST_excerpt, 5)


## ----echo=FALSE-------------
# Extraer únicamente las columnas relevantes del conjunto de datos
data_ST_excerpt <- aida::data_ST %>% 
  filter(correctness == "correct") %>% 
  select(RT, condition)

# Mostrar las primeras 5 líneas
head(data_ST_excerpt, 5)


## ----eval=FALSE, tidy=F-----
# data_ST_excerpt %>%
#   ggplot(aes(x = condition, y = RT,
#              color = condition, fill = condition)) +
#   geom_violin() +
#   theme(legend.position = "none")


## ----echo=FALSE, out.height="80%"----
data_ST_excerpt %>% 
  ggplot(aes(x = condition, y = RT, color = condition, fill = condition)) +
  geom_violin() +
  theme(legend.position = "none")


## ----eval=FALSE-------------
# data_ST_excerpt %>%
#   group_by(condition) %>%
#   summarize(mean_RT = mean(RT))


## ----echo=FALSE-------------
data_ST_excerpt %>% 
  group_by(condition) %>% 
  summarize(mean_RT = mean(RT))


## ----eval=FALSE-------------
# data_ST_excerpt %>%
#   filter(condition == "incongruent") %>%
#   pull(RT) %>%
#   mean() -
#   data_ST_excerpt %>%
#   filter(condition == "congruent") %>%
#   pull(RT) %>%
#   mean()


## ----echo=FALSE-------------
data_ST_excerpt %>% 
  filter(condition == "incongruent") %>% 
  pull(RT) %>% 
  mean() -
  data_ST_excerpt %>% 
  filter(condition == "congruent") %>% 
  pull(RT) %>% 
  mean()


## ----eval=T, results="hide"----
fit_brms_ST <- brm(
  formula = RT ~ condition,
  data = data_ST_excerpt
)


## ----eval=T-----------------
summary(fit_brms_ST)$fixed[, c("l-95% CI", "Estimate", "u-95% CI")]


## ----eval=T-----------------
data_ST_excerpt %>% 
  mutate(new_predictor = ifelse(condition == "congruent", 0, 1)) %>% 
  head(5)


## ----eval=FALSE-------------
# # Seleccionar las columnas relevantes del conjunto de datos
# data_MC_excerpt <- aida::data_MC_cleaned %>%
#   select(RT, block)
# 
# # Mostrar las primeras 5 líneas
# head(data_MC_excerpt, 5)


## ----eval=FALSE-------------
# data_MC_excerpt %>%
#   group_by(block) %>%
#   summarize(mean_RT = mean(RT))


## ----eval=FALSE, tidy=FALSE----
# data_MC_excerpt %>%
#   ggplot(aes(x = block, y = RT,
#              color = block, fill = block)) +
#   geom_violin() +
#   theme(legend.position = "none")


## ----eval=T, out.height="85%", echo=F----
data_MC_excerpt <- aida::data_MC_cleaned %>% 
  select(RT, block) 

data_MC_excerpt %>% 
  ggplot(aes(x = block, y = RT, 
             color = block, fill = block)) +
  geom_violin() +
  theme(legend.position = "none")


## ----eval=FALSE-------------
# fit_brms_mc <- brm(
#   formula = RT ~ block,
#   data = data_MC_excerpt
# )


## ----eval=FALSE-------------
# summary(fit_brms_mc)$fixed[,
#                 c("l-95% CI", "Estimate", "u-95% CI")]


## ----eval=FALSE, tidy=F-----
# data_MC_excerpt <- data_MC_excerpt %>%
#   mutate(block_reordered = factor(block,
#           levels = c("goNoGo", "reaction", "discrimination")))


## ----eval=FALSE-------------
# fit_brms_mc_reordered <- brm(
#   formula = RT ~ block_reordered,
#   data = data_MC_excerpt
# )


## ----eval=FALSE, tidy=F-----
# summary(fit_brms_mc_reordered)$fixed[,
#             c("l-95% CI", "Estimate", "u-95% CI")]


## ----echo=F, out.height="70%", out.width="55%"----
knitr::include_graphics("img/interaction_plot.png")


## ---------------------------
politeness_data <- aida::data_polite
head(politeness_data, 5)


## ----eval=F, tidy=F---------
# ggplot(politeness_data, aes(x = context,
#                             y = pitch, color = gender)) +
#   geom_boxplot() +
#   theme_minimal()


## ----echo=F, out.height="80%"----
ggplot(politeness_data, aes(x = context, 
                            y = pitch, color = gender)) +
  geom_boxplot() +
  theme_minimal()


## ----echo=T, eval=F,  tidy=F----
# fit_brms_politeness <- brm(
#   pitch ~ gender * context,
#   data = politeness_data
# )


## ----echo=F, tidy=F---------
output <- capture.output(fit_brms_politeness <- brm(
  pitch ~ gender * context,
  data = politeness_data
))


## ----echo=T,  tidy=F--------
summary(fit_brms_politeness)$fixed[, 
             c("l-95% CI", "Estimate", "u-95% CI")]


## ----echo=T,  tidy=F--------
brms::hypothesis(fit_brms_politeness, 
                 "genderM + 0.5 * genderM:contextpol < 0")


## ----echo=T,  tidy=F--------
brms::hypothesis(fit_brms_politeness, 
                 "contextpol + 0.5 * genderM:contextpol < 0")


## ----echo=T,  tidy=F--------
brms::hypothesis(fit_brms_politeness, 
                 "genderM:contextpol > 0")

