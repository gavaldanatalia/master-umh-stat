#MAPAS
library(ggplot2)
library(RColorBrewer)
library(tidyverse)
library(maps)


Estados_unidos = map_data("state") 
head(Estados_unidos) 
ggplot(Estados_unidos,aes(x=long,y=lat,group=group))+ #Mapa 
  geom_polygon(fill="white",colour="black") 

#unique(world_map$region)


#MAPA ESPAÑA
Spain_islas = c("Spain", "Canary Islands")
Spain = map_data("world", region = Spain_islas) #Coger España e Islaas Canarias

ggplot(Spain,aes(x=long,y=lat,group=group))+ #Mapa 
  geom_polygon(fill="pink",colour="red")+ #Colores
  theme(
        panel.background = element_rect(fill=NA), #Eliminar fondo
        panel.border=element_rect(fill=NA, color="yellow", size=5)) #EColor bordes

#Mapa entero
mapa_mundo = map_data("world") 
unique(mapa_mundo$region)
ggplot(mapa_mundo,aes( x= long, y = lat, group = group),) + 
  geom_polygon( fill = "black", color = "white") 

#Mapa entero tuneado
mapa_mundo = map_data("world") 
ggplot(mapa_mundo,aes( x= long, y = lat, group = group),) + 
  geom_polygon( fill = "lightblue", color = "white")+ 
  ggtitle( "Mapa Mundi") +
  theme( 
    axis.line = element_line(size = 3, colour = "pink"), #línea de los ejes 
    axis.text =  element_text(colour = "blue",size=12), #Texto ejes color y tamaño
    axis.title = element_blank(), #Sin títulos para ejes
    axis.ticks = element_line(size = 5), #Grosor puntos intervalo
    panel.background = element_rect(fill=NA),
    plot.title = element_text(family = "serif", face = "bold", 
                              color="pink",size=15,hjust=1,vjust=1,angle=-10)) 

#Mapa entero tuneado
mapa_mundo = map_data("world") 
ggplot(mapa_mundo,aes( x= long, y = lat, group = group),) + 
  geom_polygon( fill = "lightblue", color = "white")+ 
  ggtitle( "Mapa Mundi") +
  theme( 
    axis.line = element_line(size = 3, colour = "pink", linetype=3), #línea de los ejes , discontinua
    axis.text =  element_text(colour = "blue",size=12, angle=45, hjust=1, vjust=0), #Texto ejes color y tamaño
    axis.title = element_text(colour = "red", size=12, face="bold.italic", hjust=1), #Texto ejes ajustado 
    axis.ticks = element_line(size = 5), #Grosor puntos intervalo
    panel.background = element_rect(fill=NA),
    plot.title = element_text(family = "serif", face = "bold", 
                              color="pink",size=15,hjust=1,vjust=1,angle=-10))+
    labs(
      x = "long",
      y = "lat")+
    coord_fixed (xlim= c(-12,5), 
                ylim= c(35,45), 
                ratio = 1.3) #Relación entre x e y 



    
    