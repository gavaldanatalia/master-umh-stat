# PCA

library(FactoMineR)
library(dplyr)

data(decathlon)
str(decathlon)
summary(decathlon)

R <- cor(decathlon[,1:10])

library(corrplot)
corrplot(R, method = "square")

library(psych)
cortest.bartlett(R, n=nrow(decathlon))
KMO(R)

pca_1 <- prcomp(decathlon[,1:10], scale=T)
summary(pca_1)

pca_1$rotation

pca_1$sdev^2

library(factoextra)
fviz_screeplot(pca_1)

summary(pca_1)

pca_2 <- PCA(x=decathlon[,1:10],
             scale.unit = T,
             ncp = 10,
             graph = T)

fviz_contrib(pca_1,
             choice="var",
             axes = 2,
             top = 10)

fviz_contrib(pca_1,
             choice="var",
             axes = 3,
             top = 10)

var <- get_pca_var(pca_1)
var$contrib

