import UtilsDataset
import UtilsClassifier
import numpy as np
import Utils

class UtilsPSO():

    '''
    Esta classe contem atributos que sao utilizados ao longo do projeto
    e que nao pertencem a algo especifico
    '''

    INERCIA = 'inercia'
    C1 = 'c1'
    C2 = 'c2'
    NEIGHBORS = 'vizinhanca'
    ALPHA = 'alpha'

    def __init__(self, **attributes):
        self.inercia = attributes.get(self.INERCIA)
        self.c1 = attributes.get(self.C1)
        self.c2 = attributes.get(self.C2)
        self.neighbors = attributes.get(self.NEIGHBORS) # --> Este valor tem de ser passado com um valor igual ao nº de particulas, assim todos sao vizinho, deixando de existir vizinhanca, e converte-se num ótimo global
        self.alpha = attributes.get(self.ALPHA)

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

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, newAlpha):
        self.alpha = newAlpha

    def objectiveFunction(self, arrayParticleDimensions, dataset: UtilsDataset.UtilsDataset, classifier: UtilsClassifier.UtilsClassifier, alpha):

        '''
        All credits go to these documents:
         - https://pyswarms.readthedocs.io/en/development/examples/feature_subset_selection.html;
         - https://www.researchgate.net/publication/306927986_A_Feature_Selection_Method_Based_on_Modified_Binary_Coded_Ant_Colony_Optimization_Algorithm
        :param arrayParticleDimensions: Array com as posicoes binarias de uma particula, tendo em consideracao as dimensoes do problema, em causa
        :param dataset: Dataset a ser classificado
        :param classifier: Objeto com os classificadores disponiveis
        :return: custo/perda tendo em conta a funcao definida no artigo https://pyswarms.readthedocs.io/en/development/examples/feature_subset_selection.html;

        Explicacao da funcao de custo:
        TODOS OS CREDITOS DESTINAM-SE AO ARTIGO: Vieira, Mendoca, Sousa, et al. (2013)--> https://www.sciencedirect.com/science/article/abs/pii/S1568494613001361.

        f(X)=α(1−P)+(1−α)(1−Nf/Nt)
             - α: PARAMETRO PASSADO POR ARGUMENTO, ESTABELECENDO UMA RELACAO ENTRE AS RESTANTES VARIAVEIS DA FUNCAO;
             - P: REPRESENTA A ACCURACY REFERENTE A PREVISAO DO CLASSIFICADOR UTILIZADO;
             - Nf: Nº DE FEATURES RELEVANTES, QUE A PARTICULA ACREDITA QUE SAO AS NECESSARIAS À CLASSIFICACAO DO PROBLEMA;
             - Nt: Nº TOTAL DE FEATURES DO PROBLEMA;
        '''

        featuresOfProblem = dataset.getDataset().X.shape[1]

        #DEFINICAO DOS DADOS DO PROBLEMA, TENDO EM CONTA AS FEATURES QUE A PARTICULA CONSIDERA RELEVANTES PARA O PROBLEMA, SENDO EFETUADO O TREINO E PREVISAO, COM O DATASET ALTERADO (APENAS FEATURES RELEVANTES)

        #SE A PARTICULA IDENTIFICAR QUE TODAS AS FEATURES SAO RELEVANTES ESNTAO O DATASET MANTEM-SE (FEATURES TODAS A 1)
        if np.count_nonzero(arrayParticleDimensions) == 0:
            X_subset = dataset.getDataset().X
        else:
            X_subset = dataset.getDataset().X[:, arrayParticleDimensions==1] #DATASET APENAS COM AS COLUNAS REFERENTES ÀS FEATURES QUE A PARTICULA ACHA RELEVANTES, PARA CADA UMA DAS 24 LINHAS
        classifier.getSVMClassifier().fit(X_subset[range(2,22)],dataset.getDataset().Y[range(2,22)]) #TREINO TENDO EM CONSIDERACAO APENAS AS FEATURES RELEVANTES, QUE A PARTICULA INDICOU

        samplesToPredict = [X_subset[i] for i in (0,1,22,23)]
        realOutputs = [dataset.getDataset().Y[i] for i in (0,1,22,23)]

        accuracy = (classifier.getSVMClassifier().predict(samplesToPredict) == realOutputs).mean() #TESTE DO PROBLEMA, TENDO EM CONTA O TREINO EFETUADO ATRAS, E CALCULADA A ACCURACY, TENDO EM CONTA OS ACERTOS QUE EXISTIRAM
        #print(accuracy)
        #CALCULO DA FUNCAO DE CUSTO, EXPLICADA ANTERIORMENTE
        j = (alpha * (2.0 - accuracy) + (1.0 - alpha) * (1 + (X_subset.shape[1] /dataset.getDataset().X.shape[1])))

        return j

    def aplicarFuncaoObjetivoTodasParticulas(self, arrayParticleSDimensions, dataset: UtilsDataset.UtilsDataset, classifier: UtilsClassifier.UtilsClassifier, alpha):

        '''

        :param arrayParticlesDimensions: ARRAY BIDIMENSIONAL, REPRESENTANDO CADA PARTICULA, CONTENDO O VETOR DE DIMENSOES DE CADA UMA, REPRESENTANDO AS FEATURES QUE CADA UMA DELAS ACHA RELEVANTE PARA O PROBLEMA
        :param dataset: DATASET DO PROBLEMA
        :param classifier: CLASSIFICADOR A UTILIZAR NO TREINO E PREVISAO DO DATASET, TENDO EM CONTA AS FEATURES RELEVANTES, DE CADA PARTICULA
        :param alpha: HIPERPARAMETRO ALPHA, QUE ESTABELECE UMA RELACAO ENTRE OS RESTANTES PARAMETROS DA FUNCAO DE CUSTO
        :return: RESULTADO DO CUSTO DE CADA UMA DAS PARTICULAS
        '''

        nParticles = arrayParticleSDimensions.shape[0]
        j = [self.objectiveFunction(arrayParticleSDimensions[i], dataset, classifier, alpha) for i in range(nParticles)] #ARRAY QUE AGREGA O RETORNO DA EXECUCAO DA FUNCAO OBJETIVO PARA CADA UMA DAS PARTICULAS

        return np.array(j)
