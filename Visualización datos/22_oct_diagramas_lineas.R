
library(tidyverse)
library(gcookbook)

head(uspopage)

uspopage_years = aggregate(Thousands ~ Year, data = uspopage, sum)
head(uspopage_years)

ggplot(uspopage_years, aes(x = Year, y = Thousands)) +
  geom_area(colour = "black", fill = "yellow", alpha = 0.2)

ggplot(uspopage, aes(x = Year, y = Thousands, fill=AgeGroup)) +
  geom_area()

ggplot(uspopage, aes(x = Year, y = Thousands, fill=AgeGroup)) +
  geom_area(colour = "black", size = .2, alpha = 0.4)+
  scale_fill_brewer(palette="Yellows")

#Si queremos ver el crecimiento por edades en función del total, del 100%:
#position = “fill”
ggplot(uspopage, aes(x = Year, y = Thousands, fill=AgeGroup)) +
  geom_area(colour = "black", size = .2, alpha = 0.4, position = "fill")+
  scale_fill_brewer(palette="Yellows")

#+ scale_y_continuous(labels = scales::percent)
ggplot(uspopage, aes(x = Year, y = Thousands, fill=AgeGroup)) +
  geom_area(colour = "black", size = .2, alpha = 0.4, position = "fill")+
  scale_fill_brewer(palette="Yellows")+ 
  scale_y_continuous(labels = scales::percent)
