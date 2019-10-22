import UtilsDataset as uDT

class UtilsPSO:

    '''
    Esta classe contem atributos que sao utilizados ao longo do projeto
    e que nao pertencem a algo especifico
    '''

    INERCIA = 'inercia'
    C1 = 'c1'
    C2 = 'c2'
    NEIGHBORS = 'vizinhanca'
    L1DISTANCE = 'L1'
    L2DISTANCE = 'L2'
    ALPHA = 'alpha'
    DATASET = 'dataset'
    GAMMA = 'gamma'
    NEIGHBORSCLASSIFIER = 'nnclassifier'


    def __init__(self, **attributes):
        self.inercia = attributes.get(self.INERCIA)
        self.c1 = attributes.get(self.C1)
        self.c2 = attributes.get(self.C2)
        self.neighbors = attributes.get(self.NEIGHBORS) # --> Este valor tem de ser passado com um valor igual ao nº de particulas, assim todos sao vizinho, deixando de existir vizinhanca, e converte-se num ótimo global
        self.l1 = attributes.get(self.L1DISTANCE)
        self.l2 = attributes.get(self.L2DISTANCE)
        self.alpha = attributes.get(self.ALPHA)
        self.dataset = attributes.get(self.DATASET) #--> Object of UtilsDataset Type
        self.gamma = attributes.get(self.GAMMA)
        self.neighborsClassifier = attributes.get(self.NEIGHBORSCLASSIFIER)

    def getInercia(self):
        return self.inercia

    def setInercia(self, newInercia):
        self.inercia = newInercia

    def getC1(self):
        return self.c1

    def setC1(self, newC1):
        self.c1 = newC1

    def getC2(self):
        return self.c2

    def setC2(self, newC2):
        self.c2 = newC2

    def getNeighbors(self):
        return self.neighbors

    def setNeighbors(self, newNeighbors):
        self.neighbors = newNeighbors

    def getL1(self):
        return self.l1

    def setL1(self, newL1):
        self.l1 = newL1

    def getL2(self):
        return self.l2

    def setL2(self, newL2):
        self.l2 = newL2

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, newAlpha):
        self.alpha = newAlpha

    def getGamma(self):
        return self.gamma

    def setGamma(self, newGamma):
        self.gamma = newGamma

    def getNNClassifier(self):
        return self.neighborsClassifier

    def setNNClassifier(self, newNNClassifier):
        self.neighborsClassifier = newNNClassifier

    def objectiveFunction(self, posicoesBinariasParticula):



