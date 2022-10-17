import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

import pickle # to save and load the model

#incorporating SKLearn with NLTK
from nltk.classify.scikitlearn import SklearnClassifier  
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


#Combining Algo with a Voting System

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()  

all_words = []
documents = []

# j is adjective, r is adverb, and v is verb
#allowed_word_types = ["J", "R", "V"]
allowed_word_types = ["J"]

for r in short_pos.split('\n'):
    documents.append((r, "pos"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append((r, "neg"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())   

save_documents = open("documents.pickle", "wb") # saving the classifier (model) above
pickle.dump(documents, save_documents)
save_documents.close()            
    
all_words = nltk.FreqDist(all_words)    

word_features = list(all_words.keys())[:5000]
#word_features = [k[0] for k in sorted(all_words.items(),reverse=True, key=lambda x:x[1])[:3000]]

save_word_features = open("word_features.pickle", "wb") # saving the classifier (model) above
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("featuresets.pickle", "wb") # saving the classifier (model) above
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

#Positive data example:
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#Negative data example:
#training_set = featuresets[100:]
#testing_set = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(training_set)

#Applying Pickle
#classifier_f = open("naivebayes.pickle", "rb")  #loading from the saved classifier below
#classifier = pickle.load(classifier_f)
#classifier_f.close()

print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier_NB = open("naivebayes.pickle", "wb") # saving the classifier (model) above
pickle.dump(classifier, save_classifier_NB)
save_classifier_NB.close()

#MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

save_classifier_MNB = open("classifier_MNB.pickle", "wb") # saving the classifier (model) above
pickle.dump(MNB_classifier, save_classifier_MNB)
save_classifier_MNB.close()

#GaussianNB
#GNB_classifier = SklearnClassifier(GaussianNB())
#GNB_classifier.train(training_set)
#print("GNB_classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

# BernoulliNB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

save_classifier_BNB = open("classifier_BNB.pickle", "wb") # saving the classifier (model) above
pickle.dump(BNB_classifier, save_classifier_BNB)
save_classifier_BNB.close()

#LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier_LogisticRegression = open("classifier_LogisticRegression.pickle", "wb") # saving the classifier (model) above
pickle.dump(LogisticRegression_classifier, save_classifier_LogisticRegression)
save_classifier_LogisticRegression.close()

#SGDClassifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier_SGDClassifier = open("classifier_SGDClassifier.pickle", "wb") # saving the classifier (model) above
pickle.dump(SGDClassifier_classifier, save_classifier_SGDClassifier)
save_classifier_SGDClassifier.close()

#SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_classifier_SVC = open("classifier_SVC.pickle", "wb") # saving the classifier (model) above
pickle.dump(SVC_classifier, save_classifier_SVC)
save_classifier_SVC.close()

#LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier_LinearSVC = open("classifier_LinearSVC.pickle", "wb") # saving the classifier (model) above
pickle.dump(LinearSVC_classifier, save_classifier_LinearSVC)
save_classifier_LinearSVC.close()

#NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier_NuSVC = open("classifier_NuSVC.pickle", "wb") # saving the classifier (model) above
pickle.dump(NuSVC_classifier, save_classifier_NuSVC)
save_classifier_NuSVC.close()

#Combining Algo with a Voting System
voted_classifier = VoteClassifier(classifier, 
                                  MNB_classifier, 
                                  BNB_classifier, 
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier, 
                                  SVC_classifier, 
                                  LinearSVC_classifier, 
                                  NuSVC_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats)

