# Ejercicios de gráfico de sectores

# Librerías
install.packages(corrplot)
library(corrplot)

alumnos <- c(1, 5, 1, 1, 1, 3, 4, 2)
Titulaciones<- c("Matemáticas","Físico","Químico","ADE",
"Ingeniero","Estadístico","Informático","Economista")

# Número de observaciones
length(alumnos)
length(Titulaciones)

# Hacemos el gráfico base
pie(alumnos,radius=0.6)

# Ponemos etiquetas
pie(alumnos,radius=0.6,labels=Titulaciones,cex=1)

# Si queremos mostrar las frecuencias relativas
etiquetas <- paste0(round(100 * alumnos/sum(alumnos), 2), "%")
pie(alumnos, labels = etiquetas, radius=0.8)

# Si queremos mostrar las frecuencias relativas + Titulaciones
# Ejercicio 2
etiquetas_y_titulaciones <- paste0(etiquetas, " ",Titulaciones)
pie(alumnos, 
    labels = etiquetas_y_titulaciones, 
    radius=0.8, 
    col = rainbow(24),
    cex = 0.75, # tamaño de la letra
    lty=2 # raya discontinua
    )
title(main="Diagrama de sectores", 
      sub = "Este es el pie del gráfico", 
      font.main=2)
legend("topright", 
       etiquetas_y_titulaciones, 
       fill = rainbow(24),
       title = "Titulaciones",
       border="red",
       bty="n", # sin borde
       cex=0.75  # tamaño de la leta
       )

# Ejercicio 3
# Mismo ejercicio pero en 3D
install.packages("plotrix")
library(plotrix)

pie3D(alumnos,
      radius=0.8,
      col = rainbow(8),
      explode=0.1,
      labels = etiquetas_y_titulaciones, 
      labelcex = 0.85,
      border = "white",
      labelcol = "red")
par(c(2,4,2,4))
legend("topright", 
       etiquetas_y_titulaciones, 
       fill = rainbow(24),
       title = "Titulaciones",
       border="black",
       cex = 0.50,  # tamaño de la leta
       ncol = 2, # Número de columnas que tiene que tener la leyenda
       box.lty = 3, # discontinuo
       box.col = 3, # color del recuadro de la leyenda
       box.lwd = 2, # negrita del recuadro
)
title(main="Diagrama de sectores en 3D", 
      sub = "Estudiantes de máster", 
      col.main = "red",
      col.sub = "pink",
      font.main=2,
      line=3,
      )


