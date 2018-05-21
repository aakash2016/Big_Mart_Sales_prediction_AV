#DATA WRANGLING AND MANIPULATION IN R
#IMPORTING VARIOUS LIBRARIES WHICH MIGHT BE USEFUL
library(readr)  #CSV file I/O, e.g. the read_csv function 
library(plyr)   #for mapvalues() function #always install plyr package before dplyr otherwise it will give warning messages. 
library(dplyr)  #for fast data manipulation(functions like mutate,select,arrange,filter,...)
library(ggplot2)#for data visualisation
library(VIM)    #for KNN imputation
#READING THE DATASETs IN R 
BMStrain <- read.csv(file.choose())
BMStest <- read.csv(file.choose())
#A QUICK VIEW AT STRUCTURE AND SUMMARY OF BOTH DATASETS
str(BMStrain)  #BMStrain is a dataframe with 8523 obs. of  12 variables.
str(BMStest)  #BMStest is a dataframe with 8523 obs. of  11 variables.
## Notice that the variables like Item_Type are already categorical(factor, not character), so 
## there is no need to change them
## We have 7 categorical variables, 4 numerical variables and 1 integer variable.
#since in the test dataset Item_Outlet_Sales is the missing variable so it will be our response or dependent variable which we will predict by fitting suitable machine learning models on the proper set of features.  
#checking for any MISSING VALUES in train and test dataset>>best option is summary()
summary(BMStrain)
summary(BMStest)
#clearly we can see that only the variable Item_Weight has missing values in it,
#which can be imputed using KNN imputation or mean imputation methods.
#DIMENSION OF DATASETS
dim(BMStrain)
dim(BMStest)
#which contains 8523 observations and 12 variables
#summary statistics of all the variables or features, can be analyzed using the summary function from above
#Item_Idetifier,Item_Fat_Content,Item_Type,Outlet_Identifier,Outlet_Size ,Outlet_Location_Type,Outlet_Type are the variables which are categorical
#Appending the train and test dataset
## We can't append the datasets unless they have the same number of columns. Therefore, we will
## add another column(Item_outlet_Sales)(the dependent variable), to the test dataset.
BMStest$Item_Outlet_Sales <- NA

## Now, we will append the datasets, and create a new dataset named 'BMSdata'
BMSdata <- rbind(BMStrain, BMStest)
dim(BMSdata)
#Checking the missing(NA) values of the data
summary(is.na(BMSdata)) 
## is.na() returns a logical vector which indicates the number of datapoints which are missing
## TRUE indicates missing.
#SOME DATA VISUALISTION
# we will first plot and infer from the HISTOGRAMS of each continuous variable
hist(BMSdata$Item_Weight) #we can see there is no as such skewness and the frequency is more or less constant
hist(BMSdata$Item_Visibility)
#as we can see the histogram for Item_Visibility is right skewed with a long right tail we should have mean greater than the median which we can also see from summary(A102train) 
hist(BMSdata$Item_Outlet_Sales)
#also as we can see the histogram for Item_Outlet_Sales is right skewed with a long right tail we should have mean greater than the median which we can also see from summary(A102train) .
#from our shear logic the Item_Outlet_Sales and Item_Visibility should be related(the item which has more sale will obviously be more visible) which is evident from the fact that both Item_Visibility and Item_Outlet_Sales are right skewed.
library(ggplot2)
ggplot(BMSdata, aes(Outlet_Identifier, Item_Weight)) + geom_boxplot()
# we can see that only the weights of items relted to OUTO19 and OUTO27 are missing.
ggplot(BMSdata, aes(Item_Type, Item_Weight)) + geom_boxplot() 
# we can see that the weights all types of items are there. 
#similarly for,
ggplot(BMSdata, aes(Item_Fat_Content, Item_Weight)) + geom_boxplot() 
boxplot(BMSdata$Item_Visibility)
#THERE ARE SO MANY OUTLIERS FOR Item_Visibility VARIABLE which is also evident from the fact that its hist. was right skewed.
ggplot(BMSdata, aes(Item_Type,Item_Visibility)) + geom_boxplot()
# thus the dots above each item type repesent outliers in item visibility corresponding to that particular item.similarly,
ggplot(BMSdata, aes(Outlet_Identifier,Item_Visibility)) + geom_boxplot() 
# its no point to show the boxplot of Item_Visibility against the Item_Identifier.
ggplot(BMSdata, aes(Item_Outlet_Sales,Item_Visibility,col = Item_Type)) + geom_point()  
ggplot(BMSdata, aes(Item_Outlet_Sales,Item_Visibility,col = Outlet_Type)) + geom_point()

#IMPUTING THE MISSING VALUES FOR DATA SET FIRST USING KNN IMPUTATION
#FOR kNN IMPUTATION WE WILL REQUIRE "VIM" LIBRARY WHICH WE HAVE ALREADY IMPORTED
#imputing the missing values for Item_Weight.
#we can also see that Outlet_Size has observations like "_", SO WE WILL FIRST CONVERT THESE VALUES TO 'NA' VALUES SO THAT WE CAN USE,KNN IMPUTATION METHOD SINCE THIS METHOD IS ALSO APPLICABLE FOR CATEGORICAL VARIABLES..
BMSdata[BMSdata$Outlet_Size =="","Outlet_Size"] <- NA
summary(BMSdata)# all the empty values were replaced by NA

imputdata1 <- BMSdata
imputdata1 <- kNN(BMSdata, variable = c("Item_Weight","Outlet_Size"), k = 90)

# k is generally choosen to be squareroot of number of observations.
summary(imputdata1)
ncol(imputdata1) #you will see there are two additional logical columns that got created we have to remove them
imputdata1 <- subset(imputdata1,select = Item_Identifier:Item_Outlet_Sales)
summary(imputdata1)
plot(imputdata1$Item_MRP,imputdata1$Item_Weight)
BMSdata <- imputdata1
#ALSO ONE THING WHICH IS TO BE NOTED IS THAT THE VARIABLE Item_Fat_Content contains same observations with different names which need to be tackled::
BMSdata$Item_Fat_Content <- mapvalues(BMSdata$Item_Fat_Content, from = c("LF","Low Fat","low fat","Regular"), to = c("lf","lf","lf","reg"))
levels(BMSdata$Item_Fat_Content)
#lets see the boxplots of different variables.
#now we will look into the features having outliers and implement proper outlier detection technique
#we will only look onto the variables which are not categorical.
boxplot(BMSdata$Item_Weight)
#we can clearly see there are no outliers for this variable.
boxplot(BMSdata$Item_Visibility)
#BUT THERE ARE SO MANY OUTLIERS FOR Item_Visibility VARIABLE which is also evident from the fact that its hist. was right skewed.
#SO Item_Outlet_Sales SHOULD ALSO HAVE MANY OUTLIERS, LETS CHECK,,
boxplot(BMSdata$Item_Outlet_Sales)
#there are indeed many outliers
#WE GENERALLY USE THREE METHODS TO DETECT OUTLIERS 1)DISCARDING OUTLIERS 2)WINSORISATION 3)VARIABLE TRANSFORMATION
#DISCARDING OUTLIERS WILL RESULT IN REDUCTION IN NO. OF OBSERVATIONS AND VARIABLE TRANSFORMATION WILL RESULT IN VERY SMALL VALUES(skewness will still exist)
#SO THE BEST METHOD SHOULD BE WINSORISATION
#DETECTING OUTLIERS:
dataoutlier <- BMSdata 
bench <- 0.09459 + 1.5*IQR(BMSdata$Item_Visibility)
# 0.09459 is the third quartile
bench
#value comes out to be 0.1959837
dataoutlier$Item_Visibility[dataoutlier$Item_Visibility > bench] <- bench
boxplot(dataoutlier$Item_Visibility)
# as we can see all the outliers have been removed
BMSdata <- dataoutlier
# Notice that Item_Type has factors which are not food items. So, Item_Fat_Content makes nosense. 
# Hence we will add a new factor(level): "None", which will correspond to the non-food items in Item_Type.
#Adding new level in Item_Fat_Content "None".
levels(BMSdata$Item_Fat_Content) <- c(levels(BMSdata$Item_Fat_Content), "None")
## Based on Item_Type, for "health and Hygiene", "Household" and "Others",
## we will change the Item_Fat_Content factor to "None".
BMSdata[which(BMSdata$Item_Type=="Health and Hygiene"), ]$Item_Fat_Content = "None"
BMSdata[which(BMSdata$Item_Type=="Household"), ]$Item_Fat_Content = "None"
BMSdata[which(BMSdata$Item_Type=="Others"), ]$Item_Fat_Content = "None"
BMSdata$Item_Fat_Content <- as.factor(BMSdata$Item_Fat_Content)
table(BMSdata$Item_Fat_Content) # Viewing the variable
## Since we are only concerned with how old the outlet is, and not the establishment year
BMSdata$Outlet_Year <- 2018 - BMSdata$Outlet_Establishment_Year
table(BMSdata$Outlet_Year)
BMSdata$Outlet_Year <- as.factor(BMSdata$Outlet_Year)

## Visualizing Item_MRP with ggplot
library(ggplot2)
ggplot(BMSdata, aes(Item_MRP)) + geom_density(adjust = 1/5)
## It is obvious that we would be better off by converting Item_MRP to Categorical variable
#creating a new feature:
#lets create a new feature named "price" which is a categorical variable..

BMSdata$price <- "low"
BMSdata$price[BMSdata$Item_MRP >200] <-"high"
BMSdata$price[BMSdata$Item_MRP>70 & BMSdata$Item_MRP <=200] <- "medium"
summary(BMSdata)
#we can see a new variable named price is created with the property that,whenever,
#MRP <=70,  price =  "low"
#MRP >70 and MRP <= 200,  price = "medium"
#MRP >200,  price ="High"

## We are done with the data cleaning and feature engineering.
# Dividing data into train and test
BMStrain <- BMSdata[1:8523, ]
BMStest <- BMSdata[8524:14204, ]

#now we are ready to build and use our predictive model which is simply #####linear regression#####.
#one thing to be noted is that Item_Identifier and outlet_Identifier clearly have no role in predicting Item_Outlet_Sales ,so they are not significant variables.
##lets develop our linear model. 
model1 <- lm(Item_Outlet_Sales~., data = BMStrain[-c(1,7,8)])
summary(model1)
model2 <- lm(log(Item_Outlet_Sales)~., data = BMStrain[-c(1,7,8)])
summary(model2)
model3 <- lm(sqrt(Item_Outlet_Sales)~., data = BMStrain[-c(1,7,8)])
summary(model3)
par(mfrow=c(1,1))
plot(model1)

#Lets check RMSE values
library(Metrics)
rmse(BMStrain$Item_Outlet_Sales, model1$fitted.values) 
#RMSE value is 1126.891 (model which is suffering from heteroskedasticity)
rmse(BMStrain$Item_Outlet_Sales, exp(model2$fitted.values))
#New RMSE value is 1105.367 Thats Awesome! we improved a lot.

#Prediction on test_dataset
#BMStest$Item_Outlet_Sales <- predict(model2 , newdata = BMStest[-c(1,7,8,12)])
BMStest2 <- BMStest[c(1:11,13,14)]
options(warn = -1)
predicted <- predict(model2,newdata = BMStest2)
BMStest2$Item_Outlet_Sales <-exp(predicted)
Item_Identifier <- BMStest$Item_Identifier
Outlet_Identifier <- BMStest$Outlet_Identifier
output.df <- as.data.frame(Item_Identifier)
output.df$Outlet_Identifier <- Outlet_Identifier 
output.df$Item_Outlet_Sales <- exp(predicted)

write.csv(output.df , file = 'submission.csv')
