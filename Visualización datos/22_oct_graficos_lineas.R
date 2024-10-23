

library(tidyverse)
library(gcookbook)

head(tg)

ggplot(tg, aes(x=dose, y=length, fill=supp)) + geom_line()


ggplot(tg, aes(x=dose, y=length, fill=supp))+ 
  geom_line()+ 
  geom_point(size = 5)

ggplot(tg, aes(x=dose, y=length, fill=supp))+ 
  geom_point(size = 5)+
  geom_line()

ggplot(tg, aes(x=dose, y=length, fill=supp))+
  geom_line(linetype= "dotdash")+
  geom_point(size = 6)

ggplot(tg, aes(x=dose, y=length, fill=supp, shape=supp))+
  geom_line(linetype= "dotdash")+
  geom_point(size = 6)

ggplot(tg, aes(x=dose, y=length, fill=supp, shape=supp, color=supp))+
  geom_line(linetype= "dotdash")+
  geom_point(size = 6)


# lectura de fichero de datos

Temperatures <- read.csv("~/Repositorios/master-umh-stat/master-umh-stat/Visualización datos/datos/Temperatures.csv",
                              header=TRUE)

head(Temperatures)

# Graficos

ggplot(Temperatures, aes(x = Year, y = Variation)) +
  geom_ribbon(aes(ymin = Variation - conf.level95, ymax = Variation + conf.level95), alpha = 0.2) +
  geom_line()

ggplot(Temperatures, aes(x = Year, y = Variation)) +
  geom_ribbon(aes(ymin = Variation - conf.level95, ymax = Variation + conf.level95), alpha = 0.2, fill="red") +
  geom_line()

ggplot(Temperatures, aes(x = Year, y = Variation)) +
  geom_ribbon(aes(ymin = Variation - conf.level95, ymax = Variation + conf.level95), alpha = 0.2, fill="blue") +
  geom_line()

ggplot(Temperatures, aes(x = Year, y = Variation)) +
  geom_line(aes(y=Variation - conf.level95), colour = "grey50", linetype="dashed")+
  geom_line(aes(y=Variation + conf.level95), colour = "grey50", linetype="dashed")+
  geom_line()

ggplot(Temperatures, aes(x = Year, y = Variation)) +
  geom_line(aes(y=Variation - conf.level95), colour = "grey50", linetype="dotted")+
  geom_line(aes(y=Variation + conf.level95), colour = "grey50", linetype="dotted")+
  geom_line()
