# Terroristic-Text-Detection

## real-time-classification demo:

https://user-images.githubusercontent.com/31304414/124806148-37e31880-df11-11eb-9330-d587704cafca.mp4

## room for improvement:

in the demo above we see some of the best sides of our naive-bayes classifer! Unfortunately this program is not always consistent and still contains inaccuracies in it's classifications. 
Although this program is surely better than random, it definitely has room for improvement! One of the ways we can improve this program is by simply feeding it more data!
More specifically, larger quantities of 'benign' data and more specific keywords for 'hostile' data. 

## what is a naive-bayes classifier?

our classifier centers around the well-known bayes theorem:

P(label | evidence) = P(evidence | label) * P(label) / P(evidence) where label is in {'hostile', 'benign'} and evidence is in a subset of the english dictionary.

Therefore, in order to evaluate this binary classification, it follows that we must deduce P(evidence | label) and P(label) and P(evidence) from our dataset.
This can be done quite readily via calculating the frequency of the appropriate occurences and hence their probabilities.

For instance, P(evidence=[x1, x2, ..., xn]| label=[0]) = frequency(evidence=[x1, x2, ..., xn] and label=[0]) / frequency(label=[0]) where feature-vector [x1, x2, ..., xn] can be thought of as a character array/string (in fact, we import sci-kit's CountVectorizer to help with the task of converting text into vectors).

Obviously, this is a mathematical simplification of the true logic behind sci-kit's MultinomialNB classifier.
for more information, the following video helps explain the mathematical reasoning behind our classifier.
https://www.youtube.com/watch?v=lFJbZ6LVxN8
