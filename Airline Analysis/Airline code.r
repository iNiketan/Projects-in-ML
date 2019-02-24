
#Airline analysis


#_______#1.Read the data into R
airline.df <- read.csv(paste("SixAirlinesDataV2.csv", sep=""))



#####Summarize the data *******
summary(airline.df)


####Draw Box Plots / Bar Plots to visualize the distribution of each variable independently###

boxplot(FlightDuration,main="box plot",col = "yellow",horizontal = TRUE,xlab="flightduration")
boxplot(SeatsEconomy,main="box plot",col = "yellow",horizontal = TRUE,xlab="seat economy")
boxplot(SeatsPremium,main="box plot",col = "yellow",horizontal = TRUE,xlab="seat premeum")
boxplot(PitchEconomy,main="box plot",col = "yellow",horizontal = TRUE,xlab="PitchEconomy")
boxplot(PitchPremium,main="box plot",col = "yellow",horizontal = TRUE,xlab="PitchPremium")
boxplot(WidthEconomy,main="box plot",col = "yellow",horizontal = TRUE,xlab="WidthEconomy")
boxplot(WidthPremium,main="box plot",col = "yellow",horizontal = TRUE,xlab="WidthPremium")
boxplot(PriceEconomy,main="box plot",col = "yellow",horizontal = TRUE,xlab="PriceEconomy")
boxplot(PricePremium,main="box plot",col = "yellow",horizontal = TRUE,xlab="PricePremium")
boxplot(PriceRelative,main="box plot",col = "yellow",horizontal = TRUE,xlab="PriceRelative")


#####Scatter Plots#**********
#***to understand the relation between diffrant variable+++

plot(Airline,PercentPremiumSeats,main="how much airline prefer premium econommy")
scatterplotMatrix(formula = ~SeatsEconomy+SeatsPremium+PriceEconomy+PricePremium, cex=0.6, diagonal="histogram")
scatterplotMatrix(formula = ~WidthEconomy+WidthPremium+PriceEconomy+PricePremium, cex=0.6, diagonal="histogram")


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
################# Corrgram matrix###############################################################
library(corrplot)
cormat<-cor(airline.df[,c(3,6:18)])
cormat
corrplot(cormat,method="ellipse")


###########Hypothesis using a Regression Model#######################
regt<-lm(PriceRelative~SeatsEconomy+SeatsPremium+PitchEconomy+PitchPremium+WidthEconomy+WidthPremium+WidthDifference+PriceEconomy+PricePremium+PitchDifference,data = airline.df)
summary(regt)

sr<-lm(PriceRelative~Airline+PercentPremiumSeats+TravelMonth+IsInternational,data = airline.df)
summary(sr)


# T-Tests appropriate, to test Hypotheses
t.test(PriceRelative~Aircraft,var.equal=TRUE)

##data:  PriceRelative by Aircraft
#t = -2.4257, df = 456, p-value = 0.01566
#alternative hypothesis: true difference in means is not equal to 0
#95 percent confidence interval:i
#  -0.19561401 -0.02051732
#sample estimates:
# mean in group AirBus mean in group Boeing 
#0.4147682            0.5228339 
#########################################################


#8.Formulate a Regression Model:  
y = b0 + b1*x1 + b2*x2 + ..
#Think about what should 'y' be?
##$$as our consern is price difference between economy and primeum y should be PriceRelative 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Think about what could x = {x1, x2, ..} be?
##$$SeatsEconomy+SeatsPremium+PitchEconomy+PitchPremium+WidthEconomy+WidthPremium+WidthDifference+PriceEconomy+PricePremium+PitchDifference
##$$could be the on which PriceRelative depend so this could be x
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#fitting lenear regresion Model########
fitted(regt)



