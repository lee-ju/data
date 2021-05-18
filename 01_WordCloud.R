library(NLP)
library(tm)
library(SnowballC)
library(RColorBrewer)
library(wordcloud)

patentDocu = read.csv(".../01_WordCloud.csv", header = TRUE, sep = ",")

patentCont = as.matrix(patentDocu[2])
corpus <- Corpus(VectorSource(patentCont))

tdm <- TermDocumentMatrix(corpus, control = list(
					stopwords = TRUE,
					minWordLength = 3,
					removeNumbers = TRUE,
					removePunctuation = TRUE))
termMatrix <- as.matrix(tdm)
term_list <- sort(rowSums(termMatrix), decreasing = TRUE)

word_list <- data.frame(word = names(term_list), freq=term_list)
pal = brewer.pal(5, 'Set1')
wordcloud(word_list$word, word_list$freq, random.order = FALSE, colors = pal)
