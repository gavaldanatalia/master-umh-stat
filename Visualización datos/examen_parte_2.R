
############## Ejercicio 1

library(maps)
library(ggplot2)
library(tidyverse)
library(RColorBrewer)
library(dplyr)
library(ggspatial)

Europa = c(
  "Portugal", "Spain", "France"
)

datos <- map_data("world") %>%
  filter(region %in% c("Spain", "Portugal", "France"))

ciudades <- c("Madrid","Lisboa")

longitud<-c( -3.70256,-9.13333)
latitud <-c( 40.80, 38.90)

coordenadas = data.frame(
  longitud = longitud,
  latitud = latitud,
  stringAsFactors = F
)

longitud_puntos<-c( -3.70256,-9.13333)
latitud_puntos <-c( 40.4165, 38.71667)

coordenadas_puntos = data.frame(
  longitud = longitud_puntos,
  latitud = latitud_puntos,
  stringAsFactors = F
)

X_MIN <- -10  # Incluye Portugal
X_MAX <- 5    # Hasta el Mediterráneo
Y_MIN <- 35   # Hasta el sur de España
Y_MAX <- 45   # Hasta el norte de España

ggplot() + 
  geom_polygon(data = datos, aes(x = long, y = lat, group=group, fill = region),
               alpha = 0.5
  ) +
  geom_text(
            aes(x = 0, y = 43.5, label = "Algunas capitales"), 
            color = "black", size = 3) +
  geom_text(data = coordenadas, 
            aes(x = longitud, y = latitud, label = ciudades), 
            color = "blue", size = 3) +
  geom_point(data = coordenadas_puntos, 
            aes(x = longitud, y = latitud), 
            color = "blue", size = 1) +
  geom_path(data = datos, aes(x = long, y = lat, group = group),
            color = "red", linetype = "dashed", size = 0.2) +
  annotation_scale() +
  labs(title = "Peninsula Ibérica", x="", y="") +
  theme_minimal() + 
  scale_color_identity() +
  theme(
    panel.background = element_blank(),   
    panel.grid.major = element_blank(),   
    panel.grid.minor = element_blank(),   
    axis.text = element_blank(),          
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(face = "bold", color = "blue", size = 14) # Título en negrita y azul
  ) +
  coord_cartesian(xlim = c(X_MIN, X_MAX), ylim = c(Y_MIN, Y_MAX)) 


############## Ejercicio 2

# Cargar la librería ggplot2
library(ggplot2)

# Crear un dataframe con valores de x
df <- data.frame(x = seq(-20, 20, by = 0.1))  # Rango de valores de x

# Crear el gráfico con ambas funciones
ggplot(df, aes(x = x)) +
  geom_line(aes(y = exp(x), color = "y = e^x"), size = 1) +  
  geom_line(aes(y = log(x), color = "y = log(x)"), size = 1) + 
  geom_ribbon(aes(ymin = 0, ymax = log(x)), fill = "red", alpha = 0.5) +
  geom_hline(yintercept = 0, color = "black", size = 0.5, alpha = 0.5, linetype = "dashed") +  
  geom_vline(xintercept = 0, color = "black", size = 0.5, alpha = 0.5, linetype = "dashed") +  
  scale_color_manual(values = c("blue", "red")) +  
  labs(color = "Funciones", x="", y="") +
  theme_minimal() +
  coord_cartesian(ylim = c(-5, 10)) + 
  geom_point(
    aes(x = 0, y = 1), 
    color = "blue", size = 2) +
  geom_text(
    aes(x = -1.5, y = 1, label = "(0,1)"), 
    color = "black", size = 3) +
  geom_text(
    aes(x = -10, y = 4, label = "y = e^x"), 
    color = "blue", size = 3) +
  geom_text(
    aes(x = 14, y = 4, label = "y = log(x)"), 
    color = "red", size = 3) +
  geom_text(
    aes(x = -1, y = 9, label = "y"), 
    color = "black", size = 3) +
  geom_text(
    aes(x = 20, y = -0.5, label = "x"), 
    color = "black", size = 3) +
  geom_point(
    aes(x = 1, y = 0), 
    color = "red", size = 2) +
  geom_text(
    aes(x = 1, y = -0.80, label = "(0,1)"), 
    color = "black", size = 3) +
  theme(
    panel.background = element_blank(),   
    panel.grid.major = element_blank(),   
    panel.grid.minor = element_blank(),   
    axis.text = element_blank(),          
    axis.ticks = element_blank(),
    axis.line = element_blank(),
  )



