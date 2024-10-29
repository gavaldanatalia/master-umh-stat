

#install.packages("corrplot")
library(corrplot)

mcor=cor(mtcars)
mcor
round(mcor,digits=2)

corrplot(mcor)

corrplot(mcor, title = "Matriz de Correlación de mtcars", 
         mar = c(0, 0, 1, 0))

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", lower="square")

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", diag="l")

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", diag="n")

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", 
               lower="square", 
               diag="l")

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", 
               lower="square", 
               diag="l")

corrplot.mixed(mcor, title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               upper="ellipse", 
               lower="square", 
               diag="u")

corrplot.mixed(mcor, 
               title = "Matriz de Correlación de mtcars", 
               mar = c(0, 0, 1, 0),
               diag= "l",
               tl.pos = 'lt',
               order="alphabet")

# Ejercicio de clase
pal1<-colorRampPalette(c("yellow","red","orange","pink"))(n=8)
corrplot(mcor, 
         method="number", 
         title = "Matriz de Correlación de mtcars", 
         mar = c(0, 0, 1, 0),
         col=pal1, 
         tl.srt=45,
         order = 'alphabet'
         )

# Solo mostramos los valores de las correlaciones que son significativas
testRes = cor.mtest(mtcars, conf.level = 0.95)
corrplot(mcor, 
         method="number", 
         title = "Matriz de Correlación de mtcars", 
         mar = c(0, 0, 1, 0),
         tl.srt=45,
         order = 'alphabet', 
         insig ='blank', 
         p.mat = testRes$p,
)

# Ejercicio clase
corrplot(mcor, 
         method="number", 
         title = "Matriz de Correlación de mtcars", 
         mar = c(0, 0, 1, 0),
         tl.srt=45,
         tl.col=pal1,
         order = 'AOE', 
         bg="pink",
         cl.pos = "n",
         number.cex = 0.5,
         col="black",
         number.font=4,
         tl.cex=1,
         diag=FALSE
)


.libPaths()



