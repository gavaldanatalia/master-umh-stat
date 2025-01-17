
# Librerías
library(ggplot2)

setwd("/Users/jjmilla/Repositorios/master-umh-stat/master-umh-stat/Visualización datos/datos")

spain=read.csv("spain.csv")
world=read.csv("world2007.csv")

# Ejercicio 1

ggplot(spain, aes(x = year, y = lifeExp)) +
  geom_line(color = "blue") +
  labs(title = "Evolución de la Esperanza de Vida en España",
       x = "Año",
       y = "Esperanza de Vida") +
  theme_minimal()


# Ejercicio 2

ggplot(spain, aes(x = pop, y = gdp)) +
  geom_line(color = "blue") +
  labs(title = "Relación entre PIB y Población en España",
       x = "Población",
       y = "PIB") +
  theme_minimal()

# Ejercicio 3

ggplot(spain, aes(x = year)) +
  geom_line(aes(y = lifeExp, color = "Esperanza de Vida"), linewidth = 1) +
  geom_line(aes(y = gdp / 1000, color = "PIB"), linewidth = 1) +
  scale_y_continuous(
    name = "Esperanza de vida (años)",
    sec.axis = sec_axis(~.*1000, name = "PIB (sin ajustar)")
  ) +
  labs(title = "Evolución de la Esperanza de Vida y PIB en España",
       x = "Año") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal()

# Por separado
ggplot(spain, aes(x = year)) +
  geom_line(aes(y = lifeExp, color = "Esperanza de Vida"), linewidth = 1) +
  scale_color_manual(values = c("red")) +
  theme_minimal()

ggplot(spain, aes(x = year)) +
  geom_line(aes(y = gdp, color = "PIB"), linewidth = 1) +
  scale_color_manual(values = c("blue")) +
  theme_minimal()



# Ejercicio 4

ggplot(world, aes(x = continent, y = gdp)) +
  geom_boxplot(fill = "lightblue", color = "black") +
  labs(title = "Distribución del PIB por Continente (2007)",
       x = "Continente",
       y = "PIB") +
  theme_minimal()


# Ejercicio 5

ggplot(world, aes(x = reorder(country, gdp), y = gdp, fill = continent)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ continent, scales = "free_x") +
  labs(title = "Comparación del PIB por País y Continente (2007)",
       x = "País",
       y = "PIB") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Ejercicio 6

ggplot(world, aes(x = gdp, y = lifeExp)) +
  geom_point(aes(color = continent), alpha = 0.7) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  scale_x_log10() +
  labs(title = "Relación entre Esperanza de Vida y PIB",
       x = "PIB (escala logarítmica)",
       y = "Esperanza de Vida (años)") +
  theme_minimal()


# Ejercicio 7


# Crear el gráfico de densidad
ggplot(world, aes(x = gdp, fill = continent)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  labs(title = "Densidad del PIB por Continente",
       x = "PIB (escala logarítmica)",
       y = "Densidad") +
  theme_minimal()



# Crear el gráfico con un rectángulo desde 0 a 5000 en PIB con solo bordes
ggplot(world, aes(x = gdp, y = lifeExp)) +
  geom_rect(aes(xmin = 0, xmax = 5000, ymin = min(lifeExp), ymax = max(lifeExp)), 
            color = "black", fill = NA, linetype = "dashed", size = 1) +
  geom_point(alpha = 0.7) +
  labs(title = "Relación entre Esperanza de Vida y PIB",
       x = "PIB",
       y = "Esperanza de vida") +
  theme_minimal()






