##############
# Points ####
##############
data(mtcars)
mtcars$car <- rownames(mtcars)

# data_id in the aes mapping
p1 <- ggplot(mtcars, aes(wt, mpg, tooltip = car, data_id = car)) +
  geom_point_interactive(size = 4)

# data_id in the aes mapping
p2 <- ggplot(mtcars, aes(x = reorder(car, mpg), y = mpg, tooltip = car, data_id = car)) +
  geom_col_interactive() +
  coord_flip()

combined_plot <- p1 + p2 + plot_layout(ncol = 2)
combined_plot

############################
# Time series ##############
############################

# Library
library(ggplot2)
library(hrbrthemes)

# Create dummy data
data <- data.frame(
  cond = rep(c("condition_1", "condition_2"), each=10), 
  my_x = 1:100 + rnorm(100,sd=9), 
  my_y = 1:100 + rnorm(100,sd=16) 
)

# Basic scatter plot.
p1 <- ggplot(data, aes(x=my_x, y=my_y)) + 
  geom_point( color="#69b3a2") 

# with linear trend
p2 <- ggplot(data, aes(x=my_x, y=my_y)) +
  geom_point() +
  geom_smooth(method=lm , color="red", se=FALSE)

# linear trend + confidence interval
p3 <- ggplot(data, aes(x=my_x, y=my_y)) +
  geom_point() +
  geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE) 

p1 + p2 + p3 + plot_layout(ncol = 3)


############################
# Mosaicos ##############
############################
