from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, Perceptron
import pickle
import random
import sys
op = sys.argv[1]

random.seed(42)
nb = MultinomialNB()
sgd = SGDClassifier()
nb_filepath = "./models/nb.sav"
sgd_filepath = "./models/sgd.sav"
if op=='nb':
    pickle.dump(nb, open(nb_filepath,'wb'))
if op=='sgd':
    pickle.dump(sgd, open(sgd_filepath,'wb'))
if op=="all":
    pickle.dump(nb, open(nb_filepath,'wb'))
    pickle.dump(sgd, open(sgd_filepath,'wb'))
