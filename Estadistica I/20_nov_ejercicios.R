library(dplyr)
counties<-read.table("/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/counties.csv", header=T, sep=";")

# ¿Cuantos datos tenemos?
counties %>%
  count()

# ¿Cuantos datos tenemos en la variable de metro?
counties %>%
  count(metro)

# ¿Cuantos datos tenemos en la variable de state? Ordenamos
counties %>%
  count(state, sort=T)

# Los distintos estados que tenemos
counties %>%
  distinct(state)

# Columnas que necesitamos
select(counties, county, population, hispanic)

# Seleccion de columnas
counties %>%
  select(county, hispanic:pacific)

# Renombrado de columnas
counties %>%
  select(condado = county, estado = state)

# Todas las columnas menos men y women
counties %>%
  select(-men, -women)

# Todas las columnas que empiecen por income
df <- counties %>%
  select(starts_with("income"))
head(df)
str(df)

# Todas las columnas que terminen por work
df <- counties %>%
  select(ends_with("work"))
head(df)
str(df)

# Todas las columnas que contengan _
df <- counties %>%
  select(matches("_"))
head(df)
str(df)

# Todas las columnas que contengan work
df <- counties %>%
  select(matches("work"))
head(df)
str(df)

# Todas las columnas que contengan work o income
df <- counties %>%
  select(matches("work|income"))
head(df)
str(df)

# Ordenación
df <- counties %>%
  arrange(population)
head(df)
str(df)

# Ejercicio 1

df <- counties  %>%
  select(income, income_err, white) %>%
  filter(income>30000, white>85) %>%
  arrange(income)
head(df)
str(df)

# Filtros
df <- counties  %>%
  filter(state %in% c("Alabama","Alaska"))
head(df)
str(df)

df <- counties  %>%
  filter(state == "Alabama" & metro == "Nonmetro")
head(df)
str(df)

df <- counties  %>%
  filter(region %in% c("South","West") & metro == "Metro") %>%
  distinct(state)
head(df)
str(df)

# Creamos una nueva columna
df <- counties %>%
  mutate(non_white = hispanic + black + asian + pacific) %>%
  select(county, state, white, non_white) %>%
  arrange(desc(non_white))
head(df)
str(df)

# Codificar
df <- counties %>%
  mutate(non_white = recode(metro, 
                            Metro = 'Metro',
                            Nonmetro = "no_hay_metro")) %>%
      select(county, non_white, metro)
head(df)

# ifelse
df <- counties %>%
  mutate(big_city = ifelse(
    population>200000, "yes", "no"
  )) %>%
  select(county, big_city, population)
head(df)


df <- counties %>%
  transmute(population, men, women, PropMen = (men/population))
head(df)

