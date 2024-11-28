
# Base de datos
id <- 1:200
sexo <- c(rep("hombre",100),rep("mujer",100))
pais <- c(rep("Francia",30),rep("Italia",35),rep("Portugal",45),rep("Suiza",
                                                                    35),rep("Grecia",55))
edad<- c(rep("adolescente",
             30),rep("joven",40),rep("adulto",100),rep("anciano",30))
datos <- data.frame (id, sexo, pais, edad)

head(datos)

# duplicados datos
duplicated(datos)
nrow(datos[duplicated(datos),])

# si que hay duplcados si quito el id
nrow(datos[duplicated(datos[,-1]),])

# duplicados por variable
duplicated(sexo)

# duplicados por variable
duplicated(pais)

# consulta
datos %>%
  distinct(sexo, pais, edad) %>%
  count()

# Paises únicos
datos %>%
  distinct(pais)

# Ver los datos duplicados
datos %>%
  filter(duplicated(datos[,-1]))

# Ejercicio 1
library(babynames)
str(babynames)

# Nombres distintos
babynames %>%
  distinct(name)

# Cuenta los registros duplicados de la variable name.
babynames %>%
  filter(duplicated(name)) %>% count()

nrow(babynames[duplicated(babynames$name),])

# Cuenta los registros no duplicados de la variable name.
nrow(babynames[!duplicated(babynames$name),])

# Elimina del dataset los duplicados de la variable name.
babynames %>%
  filter(!duplicated(babynames[,3])) %>% count()

# Selecciona los registros duplicados de la variable name.
babynames %>%
  filter(duplicated(babynames[,3])) %>% count()


