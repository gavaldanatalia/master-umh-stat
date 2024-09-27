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
