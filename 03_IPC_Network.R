library(stringr)
library(dplyr)
library(tidyverse)
library(igraph)

ipcData <- read.csv(".../03_IPC_Network.csv", header = TRUE, sep = ',')
colnames(ipcData) <- c('Patent_No', 'IPC')

split_into_multiple <- function(column, pattern = ', ', into_prefix){
  cols <- str_split_fixed(column, pattern, n = Inf)
  cols[which(cols == "")] <- NA
  cols <- as_tibble(cols)
  m <- dim(cols)[2]
  
  names(cols) <- paste(into_prefix, 1:m, sep = "_")
  return(cols)
}

ipc_matrix <- ipcData %>%
  bind_cols(split_into_multiple(.$IPC, '[::|::]', 'IPC')) %>%
  select(Patent_No, starts_with('IPC_'))  

ipc_matrix <- ipc_matrix[,-1]
for (i in 1:nrow(ipc_matrix)){
  for (j in 1:ncol(ipc_matrix)){
    if (!is.na(ipc_matrix[i,j])){
      ipc_matrix[i,j] <- substr(str_trim(ipc_matrix[i,j], side='left'),1,4)
    }
  }
}

df <- data.frame(matrix(nrow=0, ncol=2))
for (i in 1:nrow(ipc_matrix)){
  for (j in 2:ncol(ipc_matrix)){
    if (!is.na(ipc_matrix[i,j])){
      df <- rbind(df, data.frame(ipc_matrix[i,1], ipc_matrix[i,j]))
    }
  }
}
colnames(df) <- c('from', 'to')
gdf <- graph.data.frame(df, directed = FALSE)
plot(gdf)
