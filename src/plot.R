#!/usr/bin/Rscript
filename <- "spirals_clustered.png"
data <- read.table("data/spirals_clustered.csv", header=FALSE, sep="," ,comment.char="#")
png(filename)
plot(data$V1,data$V2,col=data$V3,xlab="X",ylab="Y",main="Clustered Spirals K=2")
x <-dev.off()
