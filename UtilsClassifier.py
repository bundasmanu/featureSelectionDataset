from sklearn import svm, neighbors, linear_model

class UtilsClassifier():

    def __init__(self,gamma,vizinhos):
        self.svmClassifier = svm.SVC(gamma=gamma,)
        #self.svmClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        #self.svmClassifier = linear_model.LogisticRegression(solver='lbfgs')
        self.nearestneighborsClassifier = neighbors.KNeighborsClassifier(vizinhos, weights='uniform')

    def getSVMClassifier(self):
        return self.svmClassifier

    def getNearestNeighborsClassifier(self):
        return self.nearestneighborsClassifier