


# Paquetes

install.packages("maps")
install.packages("ggplot2")
install.packages("tidyverse")
install.packages("RColorBrewer")

# Liberías
library(maps)
library(ggplot2)
library(tidyverse)
library(RColorBrewer)
library(dplyr)


# Mapas

estados_unidos = map_data("state")
estados_unidos

ggplot(estados_unidos,aes(x=long,y=lat,group=group))+
  geom_polygon(fill="white",colour="black")


# España - Opción 1

map(regions=sov.expand("Spain"), fill=TRUE, 
    bg='white', plot=TRUE, namesonly=TRUE, col=2:2)
map.axes()
title(main = "Mapa de España", col.main = "#333333", font.main = 4)


# Map world
# Ejercicio puntuable
Europa = c(
  "Portugal", "Spain", "France", "Switzerland", "Germany",    "Austria", "UK", "Netherlands",
  "Denmark", "Poland", "Italy",  "Croatia", "Slovenia", "Hungary", "Slovakia",  "Czech republic",
  "Sweden","Finland", "Norway","Estonia", "Lithuania", "Ukraine",    "Belarus", "Romania",
  "Belgium", "Bulgaria", "Greece", "Moldova", "Latvia","Luxembourg","Serbia", "Bosnia and
Herzegovina", "North Macedonia", "Montenegro", "Albania","Ireland","Kosovo"
)

datos_europa <- map_data("world", region = Europa)

ciudades <- c("Lisboa","Madrid", "Paris","Roma")

longitud<-c( -9.1333300,-3.70256,2.3488,11.71819)

latitud <-c( 38.71667, 40.4165, 48.85341, 45.58383)

coordenadas = data.frame(
  longitud = longitud,
  latitud = latitud,
  stringAsFactors = F
)

centro_del_pais <- datos_europa %>%
  group_by(region) %>%
  summarise(long = mean(long), lat = mean(lat))

ggplot(datos_europa, aes(x=long, y=lat)) + 
  geom_polygon(aes(fill = "#D62839", color = "black", alpha = 0.7, group=group)) +
  geom_text(data = coordenadas, aes(x = longitud, y = latitud, label = ciudades), 
           color = "blue", size = 3) +
  ggtitle("Mapa", subtitle="hola") 

















ggplot() +
  geom_polygon(data = datos_europa, aes(x = long, y = lat, group = group), 
               fill = "#D62839", color = "black", alpha = 0.7) +
  geom_text(data = datos_europa, aes(x = long, y = lat, label = region), 
            size = 2, color = "black", check_overlap = TRUE) +
  geom_point(data = coordenadas, aes(x = longitud, y = latitud), 
             color = "red", size = 3) +
  geom_text(data = coordenadas, aes(x = longitud, y = latitud, label = ciudades), 
            hjust = 0.5, vjust = -1, color = "red", size = 3) +
  labs(title = "Mapa de Europa", subtitle = "Algunas capitales",
       x = "Longitud", y = "Latitud") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(color = "red", size = 12),
    axis.title.y = element_text(color = "red", size = 12)
  )


