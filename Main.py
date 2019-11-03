import pyswarms as ps
import Orange
import GEOparse
import Utils as ut
from Orange.data.pandas_compat import table_from_frame
from Bio import Geo
import UtilsDataset as udt
import UtilsFactory
import UtilsClassifier
import UtilsPSO
import numpy
import AbstractLearner, SVMLearner, KMeansLearner
import Orange.evaluation.scoring
import sys
from PyQt5.QtWidgets import *
import MainWindow
import KMeansLearner
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_argmin

def main():

    '''
        Assim era outra forma de fazer as coisas, mas o dataset nao esta a vir muito bem
        dataset = GEOparse.get_GEO(geo="GDS360", destdir="./")
        ds = ut.pandas_to_orange(dataset.table)
        print(ds.X.shape)
    '''

    data = Orange.data.Table("./datasetExplore")
    factory = UtilsFactory.UtilsFactory()
    dataset = factory.getUtil(name=ut.DATASET).getDataset(data)
    #print(dataset.dataset.X.shape)
    #print(dataset.getDataset().X)
    #print(data.domain)
    #print(data.X[0][9470:])

    '''
    Objeto Factory
    cs = UtilsFactory.UtilsFactory().getUtil(ut.PSO)
    '''

    #REFORMULACAO DO CONTEUDO DA TABLE ORANGE, PRESENTE NO OBJETO UTILS DATASET
    dataset= ut.transformMatrixDatasetInCorrectFormat(dataset)
    print(dataset.getDataset().X.shape)
    print(dataset.getDataset().Y)
    classificador = factory.getUtil(ut.CLASSIFIER).getClassifier(gamma=0.01, vizinhos=5) #CRIACAO DO CLASSIFICADOR

    #IDENTIFICACAO DO MELHOR VALOR DE K (CLUSTERS), TENDO EM CONTA UMA GAMA DE CLUSTERS E O DATASET EM ANALISE
    transposeDataset = ut.applyTranspostMatrix(dataset)
    #bestValueofK = ut.getBestValueOfK(dataset)
    kMeansObject = ut.applyClustering(300, transposeDataset)
    print(kMeansObject.getLearner().cluster_centers_.shape)
    print(kMeansObject.getLearner().labels_.shape)
    closest = pairwise_distances_argmin(kMeansObject.getLearner().cluster_centers_, transposeDataset.getDataset().X)
    print(closest)
    print(ut.getBestValuesForCluster(10,kMeansObject,transposeDataset))
    myArray = ut.createBinaryNumpyArrayWithReducedFeatures(closest, dataset)
    print(myArray)

    #OBTENCAO DAS MELHORES FEATURES DE CADA CLUSTER

    # #EXPERIMENTACAO DE CLASSIFICACAO E PREDICT
    # svmLeaner =SVMLearner.SVMLearner(gamma=0.5)
    # learner = svmLeaner.getLearner()
    # predictions = learner.fit(dataset.getDataset().X[12:18],dataset.getDataset().Y[12:18]).predict(dataset.getDataset().X)
    # print(Orange.evaluation.scoring.confusion_matrix(dataset.getDataset().Y, predictions))
    # print(ut.print_results(dataset.getDataset().Y,predictions))
    #
    # '''
    #     ABERTURA DA APLICACAO
    #     app = QApplication(sys.argv)
    #     window = MainWindow.MainWindow(dataset.getDataset())
    # '''
    #
    # #DEFINICAO DO ALGORITMO PSO
    # n_particles = 2
    # psoArgs = {UtilsPSO.UtilsPSO.INERCIA: 0.9, UtilsPSO.UtilsPSO.C1 : 1.4, UtilsPSO.UtilsPSO.C2 : 1.4, UtilsPSO.UtilsPSO.ALPHA : 0.88, UtilsPSO.UtilsPSO.NEIGHBORS : n_particles, 'p': 2} #p não é relevante, visto que todas as particulas se veem umas as outras, o p representa a distancia entre cada uma das particulas
    #
    # psoAlgorithm = factory.getUtil(ut.PSO).getPso(**psoArgs)
    # optionsPySwarms = {'c1' : psoAlgorithm.getC1(), 'c2' : psoAlgorithm.getC2(), 'w' : psoAlgorithm.getInercia(), 'k' : psoArgs.get(UtilsPSO.UtilsPSO.NEIGHBORS), 'p' : psoArgs.get('p')}
    #
    # dimensionsOfProblem = dataset.getDataset().X.shape[1] #FEATURES DO DATASET
    # initPos = ut.createArrayInitialPos(n_particles,dimensionsOfProblem,2) #COLOCANDO UM NUMERO BAIXO NO INIT_POS, PERCEBEMOS QUE AQUI A ACCURACY JA VAI ALTERANDO, POIS COMO EXISTEM POUCAS AMOSTRAS, A ACCURACY REVELA-SE QUASE SEMPRE IGUAL, MESMO TREINANDO DUAS AMOSTRAS COM ATRIBUTOS SEMELHANTES E OUTPUTS DISTINTOS
    # optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensionsOfProblem, options=optionsPySwarms, init_pos=initPos)
    # bestCost, bestPos = optimizer.optimize(psoAlgorithm.aplicarFuncaoObjetivoTodasParticulas, 2, dataset= dataset, classifier=classificador, alpha=psoAlgorithm.getAlpha())
    #
    # #CONTAGEM DE QUANTAS FEATURES SAO RELEVANTES
    # bestPos = ut.listToNumpy(bestPos)
    # newFeatures = numpy.count_nonzero(bestPos)
    # print(newFeatures)
    #
    # #CRIACAO DA COPIA
    # deepCopy = ut.createCloneOfReducedDataset(dataset,bestPos)
    # print(deepCopy.getDataset().X.shape)
    #
    # #TREINO E PREVISAO, APENAS COM AS FEATURES SELECCIONADAS
    # predictionsAfterFeatureSelection = learner.fit(deepCopy.getDataset().X[12:18],deepCopy.getDataset().Y[12:18]).predict(deepCopy.getDataset().X)
    # print(Orange.evaluation.scoring.confusion_matrix(deepCopy.getDataset().Y, predictionsAfterFeatureSelection))
    # print(ut.print_results(deepCopy.getDataset().Y,predictionsAfterFeatureSelection))
    #
    # '''
    #     FECHO DA APLICACAO
    #     sys.exit(app.exec())
    # '''

if __name__== "__main__":
    main()
