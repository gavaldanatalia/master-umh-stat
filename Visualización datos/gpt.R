library(chatgpt)

Sys.setenv(OPENAI_API_KEY = "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")

promt <- """ggplot(data) +
  geom_mosaic(aes(weight = 1, fill = c('blue', 'red', 'green'))) +
  geom_mosaic_text(title = 'Ejemplo de gráfico de mosaico con ggmosaic')
Me da error, lo puedes corregir?"""
cat(ask_chatgpt(promt))


# Crear un conjunto de datos de ejemplo
data <- data.frame(
  x = rep(c("A", "B"), each = 50),
  y = sample(c("C", "D", "E"), 100, replace = TRUE)
)

# Crear un gráfico de mosaico
ggplot(data) +
  geom_mosaic(aes(x=product(x,y), fill = x)) +
  scale_fill_manual(values = c("A" = "pink", "B" = "darkgreen")) +
  geom_mosaic_text(aes(x = product(x, y), 
                       fill=x)
  ) +
  theme_minimal()
