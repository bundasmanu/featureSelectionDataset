from sklearn import svm, neighbors

class UtilsClassifier:

    def __init__(self,gamma,vizinhos):
        self.svmClassifier = svm.SVC(gamma=gamma)
        self.nearestneighborsClassifier = neighbors.KNeighborsClassifier(vizinhos, weights='uniform')

    def getSVMClassifier(self):
        return self.svmClassifier

    def getNearestNeighborsClassifier(self):
        return self.nearestneighborsClassifier