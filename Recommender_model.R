library("readxl")
getwd()
setwd("C:/ISB_AMPBA/Term3/MLUL2/Individual_assignment")
bigbdata <- read_excel("C:/ISB_AMPBA/Term3/MLUL2/Individual_assignment/IMB575_Individual_Assignmnet.xls",sheet = "POS DATA")
head(bigbdata)
library("Matrix")
library("arules")
library("arulesViz")
library("dplyr")

#Doing group by on Member Count 

bigbdata %>% count(Member, sort = TRUE)
# Taking M38622 as this has highest number of entries 

bigbdata_m <- filter(bigbdata,bigbdata$Member=="M38622")

#Now we order()# Now we have to select only two columns from the bigbdata to make association rules

Order_desc <- bigbdata_m%>%group_by(Order)%>% summarise(Products = paste(Description, collapse = "|"))

#Removing Duplicates

Order_desc <- distinct(Order_desc)

#Writing data to a file

write.csv(Order_desc$Products,"Product1_log.csv", quote = FALSE, row.names = FALSE)
Products1_Data <- read.transactions("Product1_log.csv",sep = '|')
summary(Products1_Data)

#coerce into transactions
mytrans <- as(Products1_Data, "transactions")
inspect(mytrans)

#Running apriori rules with required support and confidence level

rules = apriori(Products1_Data, parameter=list(support=0.07, confidence=0.6,minlen=2))
my_rules <- sort(rules, by="lift",decreasing=TRUE)

head(my_rules1)

my_rules1 = as(my_rules,"data.frame")
write.csv(my_rules1,"my_rules.csv",quote=FALSE, row.names = FALSE)
head()

# Cleaning the data

library(dplyr)
library(stringr)
library(tidyr)
library(reshape2)
library(recommenderlab)

my_rules1 <- my_rules1 %>% 
  separate(rules    , c("LHS", "RHS"), "=>")

head(my_rules1)

my_rules1$RHS <- gsub("[[:punct:]]", "",my_rules1$RHS)  
my_rules1$RHS <- trimws(my_rules1$RHS, which = c("both"))


# Building the rating matrix

ratings_matrix <- as.matrix(acast(my_rules1, RHS~LHS, value.var="support"))
dim(ratings_matrix)


## recommendarlab realRatingMatrix format

R <- as(ratings_matrix , "realRatingMatrix")
R

rec = Recommender(R, method="IBCF")### recommend Model 

PRODUCT_B = "Organic Flours"   
my_rules1$RHS
my_Selected_PRODUCT <- subset(my_rules1, RHS == PRODUCT_B)
print("recommendations for you:")

prediction <- predict(rec, R[PRODUCT_B], n=2) ## you may change the model here
as(prediction, "list")

PRODUCT_B = "Gourd  Cucumber"   
my_rules1$RHS
my_Selected_PRODUCT <- subset(my_rules1, RHS == PRODUCT_B)
print("recommendations for you:")
prediction <- predict(rec, R[PRODUCT_B], n=2) ## you may change the model here
as(prediction, "list")


