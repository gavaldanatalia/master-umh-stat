
# Datos
path = "/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Estadistica/datos/spotify.csv"

data_spotify <- read.csv(path)

variables = c("track_genre", "danceability", "energy", "popularity" )
track_genre = c("reggaeton", "indie")

data <- data_spotify[(data_spotify$track_genre %in% track_genre), variables]

#
# 

arbol <- rpart( formula = track_genre ~ danceability + popularity + energy, 
                data = data, method = 'class')
rpart.plot(arbol)
