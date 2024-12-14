# Arboles de decision

library(titanic)
library(rpart)
library(rpart.plot)

data("titanic_train")
head(titanic_train) 

arbol <- rpart( formula = Survived ~ Sex + Age, 
                data = titanic_train, 
                method = 'class')
rpart.plot(arbol)

Pred_arbol<-predict(arbol, type='class')
Titanic_pred<-cbind(titanic_train,Pred_arbol) 

predict(object=arbol, newdata=data.frame(Age=5, Sex='male'),type='class') 

# Ejercicio 4

arbol <- rpart( formula = Survived ~ Sex + Age + Pclass, 
                data = titanic_train, method = 'class')
rpart.plot(arbol)

# Una mujer de 5 años y que esta en clase 3: Sería etiquetada como que sobrevive
predict(object=arbol, newdata=data.frame(Age=5, Sex='female', Pclass=3), type='class') 
