ham_or_spam <- read.csv("G:/Y_2023/R_PROGRAMMING/TEXT_ANALYTICS_TUTORIALS/sms_spam.csv")
str(ham_or_spam)
ham_or_spam$type <- as.factor(ham_or_spam$type)
which(!complete.cases(ham_or_spam))
table(ham_or_spam$type)
prop.table(table(ham_or_spam$type))
ham_or_spam$textlength <- nchar(ham_or_spam$text)
head(ham_or_spam)
hist(ham_or_spam$textlength)

library(ggplot2)

ggplot(ham_or_spam, aes(textlength, fill = type)) + geom_histogram(binwidth =5) + facet_wrap(~type)

library(lattice)

histogram(~textlength | type,data = ham_or_spam)

install.packages("tm") # tm- text mining package
library(tm)

# A vector source interprets each element of the vector x as a document
#Corpus is a collection of text document over which we would apply text mining 
# or natural language processing routines to derive inferences

sms_corpus <- Corpus(VectorSource(ham_or_spam$text)) # sms_corpus- txt mining object
print(sms_corpus)

inspect(sms_corpus[1:3]) # cannot view it directly so use inspect command

#clean up the corpus for relevant words (remove numbers,punctuation,stop words etc)

#translate all letters to lower case
sms_corpus_clean <- tm_map(sms_corpus, tolower)

#The tm package provides a function tm_map() to apply cleaning functions to an entire corpus,
#making the cleaning steps easier.
#tm_map() takes two arguments, a corpus and a cleaning function.
#For compatibility, base R and qdap functions need to be wrapped in content_transformer()
 
#remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

#remove punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#remove stop words like our, me,i,ourselves etc
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#remove unnecessary white space
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

#inspect the clean corpus
inspect(sms_corpus_clean[1:3])

#creating the document term matrix for tokenization of corpus
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
inspect(sms_dtm[1:10, 10:15])

# alternate way for cleaning the corpus and forming the document term matrix

# sms_dtm2 <- DocumentTermMatrix(sms_corpus_clean,control = list(
#   tolower= TRUE,
#   removeNumbers = TRUE,
#   stopwords = TRUE,
#   removePunctuation = TRUE,
#   stripwhitespace = TRUE
# ))

#Lets create the word cloud for each ham and spam to understand the difference
spam_cloud <- which(ham_or_spam$type == "spam")
ham_cloud <- which(ham_or_spam$type == "ham")
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_clean[ham_cloud],min.freq = 40)
wordcloud(sms_corpus_clean[spam_cloud],min.freq = 40)

#now lets build a spam filter using naive bayes classifier algorithm
#partitioning df,corpus and dtm into training and test data

ham_or_spam_train_labels <- ham_or_spam[1:4169,]$type
ham_or_spam_test_labels <- ham_or_spam[4170:5559,]$type

prop.table(table(ham_or_spam_train_labels))
prop.table(table(ham_or_spam_test_labels))

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

sms_corpus_clean_train <- sms_corpus_clean[1:4169]
sms_corpus_clean_test <- sms_corpus_clean[4170:5559]


#separate the train data into ham or spam

spam <- subset(ham_or_spam[1:4169,],type == "spam")
ham <- subset(ham_or_spam[1:4169,],type == "ham")

#finding frequent words
frequent_words <- findFreqTerms(sms_dtm_train,5)
length(frequent_words)

#let's view some of the words
frequent_words[1:10]

#creating document term matrix using frequent words
sms_freq_word_train <- sms_dtm_train[, frequent_words]
sms_freq_word_test <- sms_dtm_test[, frequent_words]

#creating a yes/no function as naive bayes classifier present or absent 
#info in each word in a message
yes_or_no <- function(x){
  y <- ifelse(x > 0, 1,0)
  y <- factor(y,levels = c(0,1),labels = c("No","Yes"))
  y
}

#applying the function on 
#sms_train and sms_test document term matrix to know the presense of word

sms_train <- apply(sms_freq_word_train,2,yes_or_no)
sms_test <- apply(sms_freq_word_test,2,yes_or_no)

head(sms_train)[1:3,1:4]
head(sms_test)[1:3,1:4]

library(e1071)

sms_classifier <- naiveBayes(sms_train, ham_or_spam_train_labels)
class(sms_classifier)

sms_test_pred <- predict(sms_classifier, newdata = sms_test)
sms_test_pred[1:5]

table(sms_test_pred,ham_or_spam_test_labels)
prop.table(table(sms_test_pred,ham_or_spam_test_labels))

#lets view it in a better way
install.packages('gmodels')
library(gmodels)
CrossTable(sms_test_pred,ham_or_spam_test_labels, prop.chisq = F,
           prop.t = F,dnn=c('Predicted', 'Actual'))







 
 





