from sklearn import svm, neighbors, linear_model
import KMeansLearner, AbstractLearner

class UtilsClassifier():

    def __init__(self,gamma,kValue):
        self.svmClassifier = svm.SVC(gamma=gamma)
        #self.svmClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        #self.svmClassifier = linear_model.LogisticRegression(solver='lbfgs')
        self.kMeans = KMeansLearner.KMeansLearner(kValue)

    def getSVMClassifier(self):
        return self.svmClassifier

    def getKMeans(self):
        return self.kMeans