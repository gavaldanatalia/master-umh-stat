library(tidyr)
library(vcd)
library(grid)

UCBAdmissions

ftable(UCBAdmissions)

# Mosaico
mosaic(~Admit+Gender+Dept, data=UCBAdmissions)

# Mosaico con colores
mosaic(~Dept+Gender+Admit, # Variables
       data=UCBAdmissions, # datos
       highlighting="Gender", # Variable que queremos colorear
       highlighting_fill=c("grey","pink"), # Colores de la variable
       highlighting_direction="left", # Si se colorea de izq a derecha o cómo
       direction = c("h", "h", "v"), # En qué posición quiero cada variable
       main = "Mi titulo del mosaico" # Titulo
) 


# Ejercicio

install.packages("devtools")
devtools::install_github("haleyjeppson/ggmosaic")
#library(ggplot2)
#library(ggmosaic)

# Crear el dataframe
datos <- data.frame(
  Edad = c("Niño", "Niño", "Niño", "Niño", "Niño", "Niño", "Niño", 
           "Adulto", "Adulto", "Adulto", "Adulto", "Adulto"),
  Dispositivo = c("Tablet", "Tablet", "Tablet", "Tablet", "Tablet", 
                  "Tv", "Tv", "Tablet", "Tablet", "Tv", "Tv", "Tv"),
  Música = c("Clásica", "Reggaetón", "Clásica", "Reggaetón", "Clásica", 
             "Reggaetón", "Clásica", "Reggaetón", "Reggaetón", 
             "Reggaetón", "Clásica", "Reggaetón")
)

datos

# Grafico

geom_mosaic(datos)

ggplot(data = datos) +
  geom_mosaic(aes(x = Edad, conds =Música, fill = Dispositivo)) 


#install.packages("ggmosaic")
#library(ggmosaic)

ggplot(data = datos) +
  geom_mosaic( aes( 
    x = product(Edad, Música, Dispositivo), 
    fill=Dispositivo), 
    divider=mosaic("h")
  ) +
  geom_mosaic_text(aes(x = product(Edad, Música, Dispositivo), 
                  fill=Dispositivo)
                   ) +
  scale_fill_manual(values = c("Tablet" = "pink", "Tv" = "darkgreen")) +
  theme_minimal()


# Mosaico con colores con un título en medio y con la leyenda que tenga los colores azul y verde
ggplot(data = datos) +
  geom_mosaic(aes(x = Edad, conds =Música, fill = Dispositivo)) +
  geom_mosaic_text(title = "Ejemplo de gráfico de mosaico con ggmosaic") +
  scale_fill_manual(values = c("Tablet" = "blue", "Tv" = "green")) +
  theme_minimal()








