from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
import pickle
import random
import sys
op = sys.argv[1]

random.seed(42)
nb = MultinomialNB()
sgd = SGDClassifier()
pa = PassiveAggressiveClassifier()
nb_filepath = "models/nb.sav"
sgd_filepath = "models/sgd.sav"
pa_filepath = "models/pa.sav"
if op == 'nb':
    pickle.dump(nb, open(nb_filepath, 'wb'))
if op == 'sgd':
    pickle.dump(sgd, open(sgd_filepath, 'wb'))
if op == 'pa':
    pickle.dump(pa, open(pa_filepath, 'wb'))
if op == "all":
    pickle.dump(nb, open(nb_filepath, 'wb'))
    pickle.dump(sgd, open(sgd_filepath, 'wb'))
    pickle.dump(pa, open(pa_filepath,'wb'))
