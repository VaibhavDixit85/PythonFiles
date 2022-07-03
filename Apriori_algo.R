library("readxl")
getwd()
setwd("C:/ISB_AMPBA/Term3/MLUL2/Individual_assignment")
bigbdata <- read_excel("C:/ISB_AMPBA/Term3/MLUL2/Individual_assignment/IMB575_Individual_Assignmnet.xls",sheet = "POS DATA")
head(bigbdata)
library("Matrix")
library("arules")
library("arulesViz")
library("dplyr")


head(bigbdata)

#Now we order()# Now we have to select only two columns from the bigbdata to make association rules

Order_desc <- bigbdata%>%group_by(Order)%>% summarise(Products = paste(Description, collapse = "|"))

#Removing Duplicates

Order_desc <- distinct(Order_desc)

#Writing data to a file

write.csv(Order_desc$Products,"Product_log.csv", quote = FALSE, row.names = FALSE)
Products_Data <- read.transactions("Product_log.csv",sep = '|')
summary(Products_Data)

#coerce into transactions
mytrans <- as(Products_Data, "transactions")
inspect(mytrans)

#Running apriori rules with required support and confidence level

rules = apriori(Products_Data, parameter=list(support=0.02, confidence=0.6,minlen=2))
my_rules <- sort(rules, by="lift",decreasing=TRUE)

inspect(my_rules)

inspect(head(sort(rules, by="lift")))


rules_subset <- subset(rules, lift > 1)
inspect(rules_subset)

