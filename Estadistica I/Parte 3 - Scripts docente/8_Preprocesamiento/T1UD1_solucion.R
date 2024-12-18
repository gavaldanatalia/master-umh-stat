library(dplyr)

#### MANIPULACION DE BASES DE DATOS

counties <- read.table("counties.csv",header = T,sep = ";")

#Visualizamos la estructura del dataset

str(counties)

#también lo podemos hacer con glimpse()

glimpse(counties)

#Contar observaciones count()

counties %>% 
  count()


#Contar observaciones de cada etiqueta de la variable metro

counties %>% 
  count(metro)

counties %>% 
  count(state)

counties %>% 
  count(state, sort = T) # Contamos por las etiquetas de una variable y ordenamos de mayor a menor

#Distinguimos los diferentes valores/etiquetas de la variable state

val1 <- counties %>% 
  distinct(state)

head(val1)

#Selección de variables en el dataset

selec1 <- select(counties,county, population, hispanic) #sin el operador %>%

head(selec1)


selec1 <- counties %>% 
  select(county, population, hispanic) #con el operador %>%

head(selec1)

#Selección de un rango de variables en el dataset usando :

selec2 <- counties %>% 
  select(county:women)

head(selec2)

#Selección y renombre de variables

selec3 <- counties %>% 
  select(estado = state, hombres = men, mujeres = women)
head(selec3)


#Solo renombrar ciertas variables

selec4 <- counties %>% 
  rename(
    estado = state,
    poblacion = population,
    hombres = men,
    mujeres = women
  )

head(selec4)

#Eliminar, borrar o deseleccionar variables (columnas)

selec5 <- counties %>% 
  select(- men, - women, - state)

str(selec5)

#Seleccionar variables en función de su nombre

# Selección de variables en función de su nombre:
# • contains(): columnas que contengan dicho término en el nombre de la variables.
# • start_with(): columnas que empiecen por dicho término.
# • ends_with(): columnas que terminen por dicho término.
# • last_col(): selecciona la última columna.

selec6<- counties %>% 
  select(contains("uction"),starts_with("income"),ends_with("work"),last_col())

head(selec6)

#Seleccionar variables por búsqueda de caracteres, por ejemplo que contenga "_"

selec7 <- counties %>% 
  select(matches("_"))

head(selec7)

# que contenga "work" or "income"

selec8 <- counties %>%
  select(matches("work|income"))

head(selec8)


#ordenar todo el dataset por una variable dada

ord1 <- counties %>% 
  arrange(population)

head(ord1)


#Seleccionar variables y ordenar de forma ascendente "arrange()"

ord2 <- counties %>%
  select(state, county, region, population) %>% 
  arrange(population)

head(ord2)


#Seleccionar variables y ordenar de forma descendente "arrange(desc())"

ord3 <- counties %>% 
  select(state, county, region, population) %>% 
  arrange(desc(population))

head(ord3)


#Filtrado de datos filter ()

filt1 <- counties %>%
  filter(income > 35000)
  

head(filt1)

#EJERCICIO 1: Selecciona los registros cuyos ingresos medios sean mayores a 35000 y
#cuyo porcentaje de población blanca sea menor al 85%. Ordena los
#registros (de mayor a menor) en función del número de empleados. Después
#de aplicar este filtro, selecciona sólo las variables state, county,
#employed, income y white.
#Filtrado de datos filter (), con selección de variables y ordenación

counties <- read.table("counties.csv",header = T,sep = ";")

filt2 <- counties %>% 
  filter(income > 35000, white < 85) %>% 
  arrange(desc(employed)) %>% 
  select(state, county, employed, income, white)

head(filt2)

#Filtrado de varios términos para una misma variable:

filt3 <- counties %>% 
  filter(state %in% c("Alabama","Alaska"))

filt3 <- counties %>% 
  filter(state == c("Alabama","Alaska"))

head(filt3)
View(filt3)

#Filtrado and (&) y  or (|)

filt4 <- counties %>% 
  filter(state == "Alabama" & metro == "Nonmetro" )


head(filt4)

#Ejercicio 2:
#Selecciona los registros de la región sur y del oeste que tengan metro.

counties %>% 
  distinct(region)

Eje2 <- counties %>%
  filter(region %in% c("South","West") & metro == "Metro" )

Eje2 <- counties %>%
  filter(region == c("South","West") & metro == "Metro" )

head(Eje2)
View(Eje2)

#Filtrado por índice de una fila slice()

counties %>% 
  slice(20:35)


#Creación de nuevas variables: mutate()

nuev1 <- counties %>% 
  mutate(not_white = hispanic + black + native + asian + pacific) %>% 
  arrange(not_white) %>% 
  select(county, state, white, not_white)

tail(nuev1)

#Renombrar etiquetas recode ()

nuev2 <- counties %>%
  mutate(metro = recode(metro,
                        Metro = "Metro",
                        Nonmetro = "No_Hay_Metro")) %>%
  select(county, state, metro)


head(nuev2)

#Con las funciones ifelse() y case_when() podemos crear nuevas variables en base a un conjunto de condiciones:

nuev3 <- counties %>%
  mutate(big_city = ifelse(
    population > 200000, "Yes","No")) %>%
  select(county, state, population, big_city)

head(nuev3)


nuev4 <- counties %>%
  mutate(too_public_work = case_when(
    public_work < 15 ~ "Low",
    public_work >= 15 ~ "High")) %>%
  select(county, state, public_work, too_public_work)

head(nuev4)


#Selección y creación de nuevas variables transmute(). Combina select() + mutate():


nuev5 <- counties %>%
  transmute(population, men, women, PropMen = round(men / population, 2),
            PropWomen = round(women / population, 2))


head(nuev5)

#LIMPIEZA DE DATOS DUPLICADOS usando la librería dplyr


library(dplyr)

#Hacemos una base de datos de ejemplo:

id <- 1:200
sexo <- c(rep("hombre",100),rep("mujer",100))
pais <- c(rep("Francia",30),rep("Italia",35),rep("Portugal",45),rep("Suiza", 35),rep("Grecia",55))
edad<- c(rep("adolescente", 30),rep("joven",40),rep("adulto",100),rep("anciano",30))

datos <- data.frame(id=id,sexo=sexo,pais=pais, edad=edad)
head(datos, n=10)

#Hacemos uso de duplicated() y de distinct() (este ?limo de la librer?a dplyr) para explorar los duplicados del dataset

#Contar duplicados

nrow(datos[duplicated(datos),])


#Contar duplicados sin tener en cuenta el id

nrow(datos[duplicated(datos[,-1]),])


#Contar los valores no duplicados (los valores distintos) sin contabilizar el id

nrow(datos[!duplicated(datos[,-1]), ])

datos %>% 
  distinct(sexo, pais, edad) %>% 
  count()


#Selecionar los elementos no repetidos de una variable, haciendo uso de la dplyr
distinct(datos,sexo,pais,edad)

#Distinguir los no duplicados haciendo uso de la libreria dplyr basado en la variable pais

datos_nd1 <- datos %>% 
  distinct(pais)

head(datos_nd1)


#Distinguir los no duplicados basado en varias variables

datos_nd2<- datos %>% 
  distinct(sexo,edad,pais) 

head(datos_nd2)



#Eliminamos duplicados de un dataframe basado en la variable edad (filas únicas)
# datos_nd4<- datos[!duplicated(datos[, 4]), ] 
# 
# head(datos_nd4)

datos_nd4 <- datos %>% 
  filter(!duplicated(datos[, 4]))

head(datos_nd4)

#Seleccionar los datos no duplicados (filas unicas)  basado en el pais
datos_nd5<- datos[!duplicated(datos$pais), ]; 

head(datos_nd5)


#Seleccionar los datos no duplicados (filas unicas) haciendo uso de unique()

# datos_nd6 <- unique(datos[, 2:4]) 
# head(datos_nd6)
# 
# datos_nd7 <- unique(datos$edad)
# head(datos_nd7)


#Seleccionar duplicados
# datos_nd5 <- datos[duplicated(datos[,2:4]), ]
# head(datos_nd5)
# str(datos_nd5)

datos_nd5 <- datos %>% 
  filter(duplicated(datos[,-1]))

str(datos_nd5)


# Ejercicio 3:
# 1. Instala la librería babynames.
# 2. Visualiza la estructura de la base de datos.
# 3. Muestra los distintos nombres de la variable name del dataset.
# 4. Cuenta los registros duplicados de la variable name.
# 5. Cuenta los registros no duplicados de la variable name.
# 6. Elimina del dataset los duplicados de la variable name.
# 7. Selecciona los registros duplicados de la variable name.

#1.Instala la libreria babynames.

library(babynames)

#2.Visualiza la estructura de la base de datos

str(babynames)

#3.Muestra los distintos nombres de la variable name del dataset

babynames1 <- babynames %>% 
  distinct(name)

head(babynames1)

#4. Cuenta los registros duplicados de la variable name

nrow(babynames[duplicated(babynames[,3]),])

babynames %>% 
  filter(duplicated(babynames[,3])) %>% 
  count()


#5. Cuenta los registros no duplicados de la variable name

nrow(babynames[!duplicated(babynames[,3]),])

babynames %>% 
  filter(!duplicated(babynames[,3])) %>% 
  count()


#6. Elimina del dataset los duplicados de la variable name

babynames2<- babynames[!duplicated(babynames[, 3]), ] 

str(babynames2)
head(babynames2)


babynames2.2 <- babynames %>% 
  filter(!duplicated(babynames[, 3]))

str(babynames2.2)


#7. Selecciona los registros duplicados de la variable name

babynames3 <- babynames[duplicated(babynames[,3]), ]

head(babynames3)
str(babynames3)


babynames3.2 <- babynames %>% 
  filter(duplicated(babynames[, 3]))

str(babynames3.2)

#4. Resumen descriptivo

#Resumen descriptivo de las variables summarize()

counties %>%
  summarize(Media = mean(population),
            Sd = sd(population),
            CV = round(sd(population) / mean(population) * 100, 2),
            Min = min(population),
            Q1 = quantile(population, 0.25),
            Q2 = quantile(population, 0.50),
            Q3 = quantile(population, 0.75),
            RIQ = IQR(population),
            Max = max(population))

#Resumen descriptivo de las variables haciendo uso de group_by(), Permite la agrupación de variables categóricas

counties %>%
  group_by(metro) %>%
  summarize(Media = mean(population))

#top_n() devuelve los n valores más altos

counties %>%
  group_by(metro) %>%
  top_n(2, population) %>%
  select(region, state, population)

#Ejercicio 4. Obtén un resumen descriptivo de los ingresos per cápita de cada una de las regiones

counties %>%
  group_by(region) %>%
  summarize(Media = mean(income_per_cap),
            Sd = sd(income_per_cap),
            CV = round(sd(income_per_cap) / mean(income_per_cap) * 100, 2),
            Min = min(income_per_cap),
            Q1 = quantile(income_per_cap, 0.25),
            Q2 = quantile(income_per_cap, 0.50),
            Q3 = quantile(income_per_cap, 0.75),
            RIQ = IQR(income_per_cap),
            Max = max(income_per_cap))

#¿Qué 3 valores coningresos per cápita son los más altos en cada una de las regiones?. 
# Visualiza de estos registros, la región, el estado, el condado y los ingresos per cápita.

counties %>%
  group_by(region) %>%
  top_n(3, income_per_cap) %>% 
  select(region, state, county, income_per_cap)

#Examen gráfico de los datos

#Cargamos la librería:

library(ggplot2)

#Gráfico de dispersión  geom_point()

ggplot(mtcars, aes(wt, mpg, color = disp)) +
  geom_point()

#Argumento referido al tamaño del dato representado. En este caso, el tamaño varía en función de la variable
#continua disp (aunque también podría tomarse un valor constante, por ejemplo, 5):

ggplot(mtcars, aes(wt, mpg, color= disp, size = disp)) +
  geom_point()

#Gráfico de suavizado geom_smooth

ggplot(mtcars, aes(wt, mpg, color = factor(cyl))) +
  geom_point() +
  geom_smooth() +
  labs(color = "cyl")

#Histograma

ggplot(mtcars, aes(mpg, fill = factor(cyl))) +
  geom_histogram(binwidth = 1,alpha = 0.4)

#Gráfico de densidades


mtcars %>%
  filter(cyl %in% c(4,8)) %>%
  ggplot(aes(x = mpg, fill = as.factor(cyl))) +
  geom_density(alpha=0.4)


#facet_grid

ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl), size = disp)) +
  geom_point() +
  scale_color_brewer(palette = "Paired") +
  facet_grid(rows = vars(gear), cols = vars(vs),
             labeller = label_both)


#EJERCICIO 5 (PARA CASA)
library(gapminder)
library(dplyr)

#1.	Visualiza los 5 primeros registros de la base de datos
datos<-gapminder 
head(datos,5)

#2.	Muestra el número de filas y de columnas del dataset, así como el nombre de las variables y su tipología.
filas<-nrow(datos) 
columnas<-ncol(datos) 
fil<-data.frame(cbind(filas,columnas)); fil
str(datos)
View(datos)
#3.	Filtrar todos los datos que sean de Perú del año 2002 y selecciona la columna país, año, esperanza de vida y población.

datos %>% 
  filter(country=="Peru", year==2002)%>% 
  select(country, year, lifeExp, pop)


#4.	Calcula la media, la desviación típica, los cuartiles, el rango intercuartílico, el mínimo, el máximo y el coeficiente de variación para la variable lifeExp en el año 2007 en cada continente. ¿En qué continente existe una mayor variabilidad en la esperanza de vida?
#¿En qué continente(s) la media ofrece un valor más representativo de la realidad?

datos %>% 
  group_by(continent)%>% 
  filter(year==2007)%>% 
  summarize(Media=mean(lifeExp), 
            Sd=sd(lifeExp), 
            Min=min(lifeExp), 
            Q1= quantile(lifeExp,0.25), 
            Q2=quantile(lifeExp,0.5), 
            Q3=quantile(lifeExp,0.75), 
            RIQ=IQR(lifeExp), 
            Max=max(lifeExp), 
            CV=sd(lifeExp)/mean(lifeExp))


#5.	¿Cuáles son los 8 países con un percentil inferior en la variable lifeExp para el año 2007? 
#Muestra el continente, el país, la esperanza de vida y el percentil (utiliza la función ntile()). 
#ntile () divide el conjunto de datos en un número específico de grupos de igual tamaño y asigna un número de grupo a cada fila

datos%>% 
  filter(year==2007)%>%
  arrange(lifeExp)%>% 
  mutate(Percentil= ntile(lifeExp,100))%>% 
  slice(1:8) %>% 
  select(country, continent,lifeExp, Percentil)


#6.	Muestra, por año, el número de habitantes totales de América, África y Europa.

a6<-datos%>% 
  group_by(year, continent)%>% 
  filter(continent %in% c("Americas", "Africa", "Europe"))%>% 
  summarize(Total=sum(pop))

View(a6)


#7.	Muestra, mediante dos gráficos de barras, el cambio en el PIB per cápita 
#de España, Reino Unido, Francia, Alemania e Italia en los años 1952 y 2007.

library(ggplot2)

a7<-datos %>% 
  filter(year %in% c(1952,2007),country %in% c("Spain", "United Kingdom", "France","Germany","Italy")) %>% 
  select(country, gdpPercap,year) 



ggplot(a7, aes(x=country, y=gdpPercap, fill=factor(year)))+ 
  geom_col(position = "dodge")+
  ylab("PIB per cápita medio")+
  xlab("País") +
  guides(fill=guide_legend(title="Años"))+ 
  ggtitle("Cambio en el PIB per cápita medio en los años 1952 y 2007")+
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 90))

#Otra forma

datos %>%
  filter(country %in% c("Spain", "United Kingdom", "France", "Germany", "Italy"),year %in% c(1952, 2007)) %>%
  group_by(country, year) %>%
  summarize(meanGdpPercap = mean(gdpPercap)) %>%
  ggplot(aes(x = country, y = meanGdpPercap, fill = country)) +
  geom_col() +
  facet_wrap(~ year) + 
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 90)) +
  xlab("Año") +
  ylab("PIB per capita (medio)") 


#8.	Realiza un histograma, por continente, para la esperanza de vida.

ggplot(datos) +
  geom_histogram(aes(x = lifeExp, fill=continent),colour = "black", bins=30)+ 
  scale_x_continuous()+ 
  facet_grid(continent ~ .) + 
  theme(legend.position='none')

#otra forma

datos %>%
  ggplot(aes(x = lifeExp)) +
  geom_histogram(aes(y = (..count..)/sum(..count..),fill = continent)) +
  scale_y_continuous("Porcentaje",labels=scales::percent) +
  facet_wrap(~ continent) +
  theme(legend.position = "bottom") +
  xlab("Esperanza de vida") +
  labs(fill = "Pais")

#9.	Crea un gráfico de dispersión para las variables gdpPercap y lifeExp para los datos 
#del año 2007. El color del punto debe variar en función del continente al que 
#pertenezca dicho país y el tamaño del punto debe ser 
#proporcional al número de habitantes de dicho país. Aplica el tema theme_bw().

datos %>%
  filter(year == 2007) %>%
  ggplot(aes(x = gdpPercap,y = lifeExp, colour = continent)) +
  geom_point(aes(size = pop)) +
  xlab("GDP per capita") +
  ylab("Esperanza de vida") +
  labs(color = "Continente", size = "Poblacion") +
  theme_bw()


#Otra forma:


a9<-datos%>% 
  filter(year==2007)%>% 
  select(continent, gdpPercap, lifeExp, pop) 

ggplot(a9, aes(x=gdpPercap, y=lifeExp,colour = continent))+ 
  geom_point(aes(size = pop))+ 
  ggtitle("Gráfico de dispersión para el PIB per Cápita Medio \n y la Esperanza de Vida en el 2007")+ 
  xlab("PIB per Cápita Medio")+ 
  ylab("Esperanza de Vida")+ 
  theme_bw()


#10.	Muestra, mediante gráficos de cajas, la esperanza de vida de los continentes.

ggplot(datos, aes(y=lifeExp, colour=continent))+ 
  geom_boxplot()+ 
  facet_grid(. ~ continent )+ 
  theme(legend.position='none')+ 
  ylab("Esperanza de Vida")



