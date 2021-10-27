##### CIPC (2021. 10. 27)
##################
## Read Dataset ##
##################
raw_Data = read.csv('C:/Temp/cipc_data.csv', stringsAsFactors=FALSE)
dim(raw_Data)

###############
## Wordcloud ##
###############
################# 한글 Wordcloud
# Check
# JDK >= 12
# 경로체크; C:\Program Files\Java\jdk-12
# Sys.setenv(JAVA_HOME="C:/Program Files/Java/jdk-12")
# 또는 시스템 환경변수에 JAVA 추가됐는지 확인

install.packages("multilinguer")
library(multilinguer)
install_jdk()
install.packages(c('stringr',
                   'hash',
                   'tau',
                   'Sejong',
                   'RSQLite',
                   'devtools'), type = "binary")
install.packages("remotes")
remotes::install_github('haven-jeon/KoNLP',
                        upgrade = "never",
                        INSTALL_opts=c("--no-multiarch"))
library(KoNLP)

#install.packages("rJava")
library(rJava)
library(KoNLP)

#install.packages("memoise")
library(memoise)

#install.packages("dplyr")
library(dplyr)
library(wordcloud)
library(RColorBrewer)

#useNIADic() # Console 창에 입력창 나올 때마다 "1" 입력
useSejongDic()

head(raw_Data[raw_Data$V2==c('KR', 'JP'),]$V3)[1]
# 한글 깨지면: 
#Tools > Code > Saving > Serialization > Default text encoding = UTF-8

library(stringr)
#Text = str_replace_all(Data$Text, "\\W", "")
Text = raw_Data$V4 # 요약문

# 전처리
Text = sapply(Text, extractNoun, USE.NAMES=FALSE)
Text = unlist(Text)
Text = gsub("【과제】", "", Text) # 특정 문자 대체
Text = gsub("【해결수단】", "", Text) # 특정 문자 대체
Text = gsub("【선택도】", "", Text) # 특정 문자 대체
Text = gsub("[~!@#$%&*()_+=?<>]", "", Text) # 정규표현식: 특수문자 삭제
Text = gsub("[ㄱ-ㅎ]", "", Text) # 정규표현식: 자음만 있는 경우 삭제
Text = gsub("[A-Z]", "", Text) # 정규표현식: 영어 대문자 삭제
Text = gsub("[a-z]", "", Text) # 정규표현식: 영어 소문자 삭제
Text = gsub("[\\d+]", "", Text) # 정규표현식: 숫자 삭제
Text = gsub("중공\\S*", "중공", Text) # 정규표현식: 특정단어 + 조사삭제
Text = gsub("시스템", "", Text) # 너무 많이 나오는 "시스템" 삭제
Text = Filter(function(x){nchar(x)>=3},
              Text)
head(Text)
#Noun = extractNoun(Text)

num_Words = table(Text)
num_Words_df = as.data.frame(num_Words, stringAsFactors=FALSE)
colnames(num_Words_df) = c("Var1"="Words", "Freq"="Frequency")

pal = brewer.pal(8, "Dark2")
wordcloud(words=num_Words_df$Words,
          freq=num_Words_df$Frequency,
          min.freq=1,
          max.words=80,
          random.order=FALSE,
          scale=c(3, 0.5),
          colors=pal)

# 명사 추가하는 법
mergeUserDic(data.frame(c("노잼"), "ncn"))

################# 영어 Wordcloud
#install.packages("SnowballC")
#install.packages("tm")
#install.packages("wordcloud")
#install.packages("RColorBrewer")

library(SnowballC)
library(tm)
library(wordcloud)
library(RColorBrewer)

eng_Text = raw_Data[raw_Data$V2==c('US', 'EP'), ]$V4
dataCorpus = Corpus(VectorSource(eng_Text))
dataCorpus = tm_map(dataCorpus, stripWhitespace)
dataCorpus = tm_map(dataCorpus, tolower)
dataCorpus = tm_map(dataCorpus, removeNumbers)
dataCorpus = tm_map(dataCorpus, removePunctuation)
dataCorpus = tm_map(dataCorpus, removeWords, stopwords("english"))

tdm = TermDocumentMatrix(dataCorpus)
tdm_matrix = as.matrix(tdm)

wordFreq = sort(rowSums(tdm_matrix), decreasing=TRUE)
barplot(wordFreq[1:10], las=2) # las=1이면 라벨을 수평축과 평행
# las=2이면 라벨을 수평축과 수직


pal = brewer.pal(8, "Dark2")

set.seed(12)
wordcloud(words=names(wordFreq), freq=wordFreq, scale=c(3, 0.1),
          min.freq=5, colors=pal, random.order=FALSE)

################
## Regression ##
################
app_date = raw_Data$V11 # 출원일자
app_year = substr(app_date, 1, 4)
year_freq = table(app_year)

X = as.integer(names(year_freq))
Y = as.integer(year_freq)

plot(Y~X)
M = lm(Y~X)
abline(M, col="red")
summary(M)
