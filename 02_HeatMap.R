library(stringr)
library(dplyr)
library(pheatmap)

patentDataset <- read.csv('.../02_HeatMap.csv', header = TRUE, sep = ',')
colnames(patentDataset) <- c('Patent_No', 'IPC_all', 'Inventor', 'Fm_patent', 'Forward', 'Fm_Country')

count <- data.frame(matrix(nrow=0, ncol=3))
for (i in (1:nrow(patentDataset))){
  count_inventor = str_count(patentDataset$Inventor[i], "[::|::]")
  count_IPC = str_count(patentDataset$IPC_all[i], "[::|::]")
  count_FmPatent = str_count(patentDataset$Fm_patent[i], "[::|::]")
  
  df = data.frame(count_inventor, count_IPC, count_FmPatent)
  count = rbind(count, df)
}
count = count+1

row_name <- patentDataset[,1]
newDataset <- cbind(patentDataset, count)[,-(1:4)]
row.names(newDataset) <- (row_name)

newDataset[is.na(newDataset)] <- 0
newDataset$Fm_Country <- as.numeric(newDataset$Fm_Country)

newDataset <- as.matrix(newDataset)
pheatmap(newDataset[1:10,], scale='none', legend = TRUE)
