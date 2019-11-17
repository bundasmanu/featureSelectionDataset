from  AbstractLearner import AbstractLearner
import Orange.classification
from sklearn.svm import SVC

#REPRRESENTA O WIDGET DO ORANGE --> SVM
#DOCUMENTACAO https://orange.readthedocs.io/en/latest/reference/rst/Orange.classification.svm.html

class SVMLearner(AbstractLearner):

    def __init__(self, gamma):
        self.svmLeaner = SVC(kernel='rbf', gamma=gamma)

    def getLearner(self):
        return self.svmLeaner