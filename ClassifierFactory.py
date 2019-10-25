import UtilsClassifier

class ClassifierFactory:

    def getClassifier(self,gamma,vizinhos):
        return UtilsClassifier.UtilsClassifier(gamma,vizinhos)