# Mi primer día con R

# Set de mi directorio de trabajo
setwd("/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Visualización datos")

# Ejercicio 1
# a) Es una variable discreta, cualitativa o categórica

# b) Dataframe
redes_sociales=c("instagram","tiktok","twitter","facebook","otras")
frecuencia_absoluta=c(19,14,10,4,3)
frecuencia_relativa=c(0.38,0.28,0.2,0.08,0.06)
datos = data.frame(redes_sociales, frecuencia_absoluta, frecuencia_relativa)

# c)
# Gráfico de barras
barplot(datos$frecuencia_absoluta)

# Gráfico de barras con nombres
barplot(datos$frecuencia_absoluta, name=datos$redes_sociales)

# cex cambia los tamaños (de los ejes y de los nombres)
barplot(datos$frecuencia_absoluta, name=datos$redes_sociales, cex.axis = 0.7, cex.names = 0.7)

# Resto de ejercicios va a subirlos al campus. Son más variaciones de lo mismo, gráficos de barras.

## Nuevo datasert de datos
# Ejercicio 4
redes_sociales=c("instagram","tiktok","twitter","facebook","otras")
frecuencia_absoluta=c(4,2,16,55,3)
datos = data.frame(redes_sociales, frecuencia_absoluta)

# Row names
rownames(datos)=datos$redes_sociales
datos = datos[,-1]

# Matrix (lo pasamos para matrix)
datos=as.matrix(datos)

# Barplot
barplot(datos,
        beside = TRUE,
        col=rainbow(5),
        legend.text = row.names(datos))


#Ejerecicio 5: Construye dos diagramas de sectores como los siguientes (los colores intenta que se parezcan, pero no hace falta que sean exactamente los mismos):

#a
# Datos
redes.sociales <- c("instagram", "tiktok", "twitter", "facebook", "otras")
jovenes <- c(19, 14, 10, 4, 3)
mayores <- c(4, 2, 16, 25, 3)

# Colores personalizados
colores <- c("darkgreen", "orange", "blue", "pink", "lightgreen")

# Función para crear diagrama de sectores mejorado
pie(jovenes, 
    labels = paste0(redes.sociales, "\n"),
    col = colores,
    main = titulo,
    border = "white",
    radius = 1,
    density = 30,
    cex = 0.8,
    init.angle = 90)

#b
# install.packages("plotrix")
library(plotrix)

# Datos
redes.sociales <- c("instagram", "tiktok", "twitter", "facebook", "otras")
jovenes <- c(19, 14, 10, 4, 3)
mayores <- c(4, 2, 16, 25, 3)

# Colores personalizados
colores2 <- c("tomato4", "orange", "peachpuff", "turquoise", "blue")

# Función para crear diagrama de sectores mejorado
pie3D(jovenes, 
      labels = paste0(redes.sociales, "\n"),
      col = colores2,
      border = "white",
      radius = 1,
      density = 30,
      cex = 0.8,
      init.angle = 90,
      explode = 0.2)
