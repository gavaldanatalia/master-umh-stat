
######################################################
######### Working directory and libraries ############
######################################################

# set working directory
setwd("/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/TFM")

# check path or working directory
getwd()

# load libraries
library(readxl)
library(ggplot2)
library(dplyr)
library(magrittr)
library(lubridate)
library(stringr)

######################################################
################## Load data #########################
######################################################
  
# load data 
df <- read_excel("bbdd.xls", sheet = "Negocio Proporcional")

# check data
head(df)

# check number of rows and columns
dim(df)

######################################################
################## Transformations ###################
######################################################

# change column names
new_names <- c(
  "centro_suscripcion", "filial", "gestion", "unidad_suscr", "unidad_gestion",
  "codigo_facultativo", "estado", "gestion_2", "suscr", "ejercicio",
  "fecha_efecto", "fecha_apertura_chantier", "fecha_inicio_cobertura",
  "fecha_vencimiento", "asegurado", "riesgo", "producto", "top", "sector",
  "tipo_riesgo", "compania_cedente", "pais_cedente", "grupo_intermediario",
  "intermediario", "division", "tipo_contrato", "num_linea", "sumas_aseguradas",
  "portee", "prioridad_franquicia", "tasa_prima", "aliment_global",
  "parte_aceptada", "parte_scor", "aliment_scor", "sumas_aseguradas_scor",
  "monto_limite_scor", "ecg", "divisa_ecg", "mercado", "pais_analisis", "trigger"
)
colnames(df) <- new_names

# change the data type
df$fecha_efecto <- as.Date(df$fecha_efecto, format = "%Y-%m-%d")
df$fecha_apertura_chantier <- as.Date(df$fecha_apertura_chantier, format = "%Y-%m-%d") 
df$fecha_inicio_cobertura <- as.Date(df$fecha_inicio_cobertura, format = "%Y-%m-%d")
df$fecha_vencimiento <- as.Date(df$fecha_vencimiento, format = "%Y-%m-%d")

# Crear una nueva variable 
df$fecha_efecto_trimestre <- quarter(df$fecha_efecto)
df$fecha_efecto_mes <- month(df$fecha_efecto)

# see the data type of data
str(df)

# Get class balance
class_balance <- function(df, columna) {
  df %>%
    count(!!sym(columna)) %>%
    ggplot(aes(x = reorder(!!sym(columna), -n), y = n)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    theme_minimal() +
    labs(title = paste("Distribución de", columna),
         x = columna,
         y = "Frecuencia") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Class balance by column
class_balance(df, "tipo_riesgo")
class_balance(df, "compania_cedente")
class_balance(df, "centro_suscripcion")

######################################################
################## Outliers ##########################
######################################################

# Outliers in tasa_prima
df %>%
  ggplot(aes(x = tasa_prima)) +
  geom_boxplot() +
  theme_minimal()

# Get outliers with IQR
q1 = quantile(df$tasa_prima, 0.25)
q3 = quantile(df$tasa_prima, 0.75)
iqr = q3 - q1
lower_bound = q1 - 3 * iqr
upper_bound = q3 + 3 * iqr

# Drop outliers in df
df_no_outliers <- df %>%
  filter(tasa_prima > lower_bound & tasa_prima < upper_bound)

dim(df_no_outliers)

# df without outliers
df_no_outliers %>%
  ggplot(aes(x = tasa_prima)) +
  geom_boxplot() +
  theme_minimal()

# summarise data
df_no_outliers %>%
  summarise(
    mean_tasa_prima = mean(tasa_prima),
    sd_tasa_prima = sd(tasa_prima),
    min_tasa_prima = min(tasa_prima),
    q1_tasa_prima = quantile(tasa_prima, 0.25),
    median_tasa_prima = median(tasa_prima),
    q3_tasa_prima = quantile(tasa_prima, 0.75),
    max_tasa_prima = max(tasa_prima)
  )

# df without outliers
df <- df_no_outliers

######################################################
################## Graphs ############################
######################################################

# Histograma de la variable objetivo para ver su distribución
ggplot(df, aes(x = tasa_prima)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribución", x = "Tasa de prima", y = "Frecuencia")


##### hist of rows by year of fecha_efecto
ggplot(df, aes(x = fecha_efecto)) + geom_histogram(binwidth = 365)


##### tasa_prima con variables continuas #####
# Graph sumas_aseguradas and tasa_prima 
ggplot(df, aes(x = sumas_aseguradas, y = tasa_prima)) + geom_point()

# Graph sumas_aseguradas and tasa_prima 
ggplot(df, aes(x = sumas_aseguradas_scor, y = tasa_prima)) + geom_point()

# Graph sumas_aseguradas and tasa_prima 
ggplot(df, aes(x = sumas_aseguradas_scor, y = sumas_aseguradas)) + geom_point() + geom_smooth(method = "lm")



##### tasa_prima con variables de tiempo #####
# Graph fecha_efecto and tasa_prima 
ggplot(df, aes(x = fecha_efecto, y = tasa_prima)) + geom_point()

# Graph fecha_efecto and tasa_prima, with tipo_riesgo as color
ggplot(df, aes(x = fecha_efecto, y = tasa_prima, color = tipo_riesgo)) + geom_point()



##### tasa_prima con variables categoricas #####
# Graph tasa_prima in a boxplot by fecha_efecto_trimestre and month
ggplot(df, aes(x = factor(fecha_efecto_trimestre), y = tasa_prima)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(df, aes(x = factor(fecha_efecto_mes), y = tasa_prima)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Graph tasa_prima in a boxplot by tipo_riesgo
ggplot(df, aes(x = tipo_riesgo, y = tasa_prima)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Graph tasa_prima in a boxplot by compania_cedente 
ggplot(df, aes(x = compania_cedente, y = tasa_prima)) + geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Crear la nueva variable "CompaniaAgrupada" eliminando etiquetas finales como " A", " B", etc.
data <- data %>%
  mutate(CompaniaAgrupada = case_when(
    str_detect(compania_cedente, regex("MAPFRE", ignore_case = TRUE)) ~ "MAPFRE",
    str_detect(compania_cedente, regex("AXA", ignore_case = TRUE)) ~ "AXA",
    str_detect(compania_cedente, regex("SCOR", ignore_case = TRUE)) ~ "SCOR",
    str_detect(compania_cedente, regex("ALLIANZ", ignore_case = TRUE)) ~ "ALLIANZ",
    str_detect(compania_cedente, regex("REALE", ignore_case = TRUE)) ~ "REALE",
    str_detect(compania_cedente, regex("UAP", ignore_case = TRUE)) ~ "UAP",
    str_detect(compania_cedente, regex("ZURICH", ignore_case = TRUE)) ~ "ZURICH",
    str_detect(compania_cedente, regex("AGF", ignore_case = TRUE)) ~ "AGF",
    # Si no se cumple ninguna condición, se mantiene el valor original
    TRUE ~ compania_cedente
  ))
ggplot(data, aes(x = fecha_efecto, y = tasa_prima, color = CompaniaAgrupada)) + 
  geom_point()

# Ver una muestra para confirmar
head(data[, c("compania_cedente", "CompaniaAgrupada")])



######################################################
################## Notas #############################
######################################################

# Preguntas hoy 19 de feb
# Idea 1
# Idea: Tratar de predecir la tasa de prima para un solicitante nuevo (en función de unas caracteristicas, predecir qué tasa de riesgo se le va a aplicar)
# Problema supervisado, la tasa de prima es conocida para un histórico.

# Consideraciones
# ¿Quitamos los outliers? De manera que solo dejemos tasas>0 y tasas<25 aprox?
# ¿Cómo jugamos con las clases que no están balanceadas?
# ¿Porcentaje reasegurado? ¿Qué es? ¿sumas_aseguradas_scor / sumas_aseguradas?


