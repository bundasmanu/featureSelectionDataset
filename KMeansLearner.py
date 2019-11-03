from AbstractLearner import AbstractLearner
from sklearn.cluster import KMeans

#REPRRESENTA O WIDGET DO ORANGE --> KNN

class KMeansLearner(AbstractLearner):

    def __init__(self, k):
        self.kmeans = KMeans(k, random_state=10, max_iter=200)

    def getLearner(self):
        return self.kmeans