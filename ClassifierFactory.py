import UtilsClassifier

class ClassifierFactory:

    def getDataset(self,gamma,vizinhos):
        return UtilsClassifier.UtilsClassifier(gamma,vizinhos)