# Ejercico entregable
# Natalia Gavaldá Lizán
# install.packages("babynames")

# Librería
library(babynames)
library(ggplot2)

# Dataset
head(babynames)

# Nombres elegidos: "Mary" y "James"
selected_names <- c("Mary", "James")
data_filtered <- babynames[babynames$name %in% selected_names, ]

# Crear el gráfico de líneas para la evolución en el tiempo de ambos nombres
ggplot(data_filtered, aes(x = year, y = n, color = name)) +
  geom_line(size = 1) +
  labs(title = "Evolución de los nombres Mary y James en el tiempo",
       x = "Año", 
       y = "Número de bebes") +
  theme_minimal()

