# below are the external sources used to generate the keywords I wanted: 
# https://www.dailymail.co.uk/news/article-2150281/REVEALED-Hundreds-words-avoid-using-online-dont-want-government-spying-you.html
# https://relatedwords.org/relatedto/horror
# https://randomtextgenerator.com/


# DEPENDENCIES
##############
import os
import io
import numpy
from pandas import DataFrame


# FUNCTIONS
###########
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                lines.append(line)
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


# INITIALIZE DATA
#################
data = DataFrame({'message': [], 'class': []})
data = data.append(dataFrameFromDirectory(
    'C:/Users/abiel/Desktop/python/data/hostile', 'hostile'))
data = data.append(dataFrameFromDirectory(
    'C:/Users/abiel/Desktop/python/data/benign', 'benign'))


# TRAINING
##########
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1) create instance of CountVectorizer object
vectorizer = CountVectorizer()
# 2) CountVectorizer creates a term 'count' matrix - document term matrix
term_matrix = vectorizer.fit_transform(data['message'].values)
# 3) create instance of MultinomialNB object
classifier = MultinomialNB()
# 4) grab y-variable - iterable
targets = data['class'].values

# 5) MultinomialNB fits document-term matrix to y-value
classifier.fit(term_matrix, targets)
# effectively maps the count-matrix of each term to its' associated y-value,
#and saves that association as an internal state object. This model we have
#can now be applied to test data.


# PREDICTION
############
isDone = False
while not isDone:
    _input = input("TEST (\"EXT\" to exit): ")
    if _input == "EXT":
        isDone = True
        break
    test = [_input]

    test_matrix = vectorizer.transform(test)
    prediction = classifier.predict(test_matrix)
    print(prediction)

    