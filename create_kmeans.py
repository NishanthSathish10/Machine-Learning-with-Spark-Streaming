import pickle
import random
from sklearn.cluster import MiniBatchKMeans

random.seed(42)

km = MiniBatchKMeans(n_clusters = 2,batch_size=5000)

pickle.dump(km, open('km.sav', 'wb'))

