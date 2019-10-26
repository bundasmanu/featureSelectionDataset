from AbstractLearner import AbstractLearner
import Orange.classification, Orange.classification.knn

#REPRRESENTA O WIDGET DO ORANGE --> KNN

class KNNLearner(AbstractLearner):

    def __init__(self, vizinhos):
        super.__init__()
        self.knnLeaner = Orange.classification.knn.KNNLearner(n_neighbors=vizinhos)

    def getLearner(self):
        return self.knnLeaner