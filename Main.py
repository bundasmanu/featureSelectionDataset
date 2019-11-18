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
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import random
from sklearn.model_selection import GridSearchCV


'''
    OPINIAO PESSOAL DA UTILIZACAO DE CLUSTERING NA SELECCAO DE FEATURES
    https://www.quora.com/How-can-I-use-k-means-for-feature-extraction
'''

def main():

    '''
        Assim era outra forma de fazer as coisas, mas o dataset nao esta a vir muito bem
        dataset = GEOparse.get_GEO(geo="GDS360", destdir="./")
        ds = ut.pandas_to_orange(dataset.table)
        print(ds.X.shape)
    '''

    '''
        DEFINICAO INICIAL DO DATASET E DO CLASSIFICADOR UTILIZADO PARA TREINO E PREVISAO,
        E DE OUTROS DADOS RELEVANTES
    '''

    data = Orange.data.Table("./datasetExplore")
    factory = UtilsFactory.UtilsFactory()
    dataset = factory.getUtil(name=ut.DATASET).getDataset(data)
    #print(dataset.dataset.X.shape)
    #print(dataset.getDataset().X)
    #print(data.domain)
    #print(data.X[0][9470:])

    #REFORMULACAO DO CONTEUDO DA TABLE ORANGE, PRESENTE NO OBJETO UTILS DATASET
    dataset= ut.transformMatrixDatasetInCorrectFormat(dataset)
    #print(dataset.getDataset().X.shape)
    #print(dataset.getDataset().Y)
    classificador = factory.getUtil(ut.CLASSIFIER).getClassifier(gamma=0.01, vizinhos=5) #CRIACAO DO CLASSIFICADOR

    examplesPredict = [0,1,22,23]
    examplesTraining = range(2,22)

    svmLeaner1 =SVMLearner.SVMLearner(gamma=0.5)
    learner = svmLeaner1.getLearner()

    '''
        TREINO E PREVISAO DO DATASET ORIGINAL
    '''

    #APLICACAO DE CROSS VALIDATION
    dataset = ut.applyMinMaxScaler(dataset)
    kFold = KFold(n_splits=4, shuffle=True)
    print("Score e Previsoes Iniciais - Cross Validation\n")
    result = cross_val_score(learner, dataset.getDataset().X, dataset.getDataset().Y, cv=kFold, scoring='r2')
    print("Media Score:\t",result.mean())
    predictions = cross_val_predict(learner, dataset.getDataset().X, dataset.getDataset().Y, cv=3)
    print("Prediction:\t",predictions)
    print("\nSVM\n")
    learner.fit(dataset.getDataset().X[examplesTraining], dataset.getDataset().Y[examplesTraining])
    listSamplesPredict = ut.getSpecificSamples(dataset, examplesPredict)
    predictions = learner.predict(listSamplesPredict)
    realValuesPredict = ut.getSpecificOutputsFromDataset(dataset, examplesPredict)
    print(Orange.evaluation.scoring.confusion_matrix(realValuesPredict, predictions))
    print(ut.print_results(realValuesPredict,predictions))

    #EXPERIMENTACAO DE CLASSIFICACAO E PREDICT
    #listSamplesPredict = ut.getSpecificSamples(dataset, examplesPredict)
    #predictions = learner.fit(dataset.getDataset().X[examplesTraining],dataset.getDataset().Y[examplesTraining]).predict(listSamplesPredict)
    #realValuesPredict = ut.getSpecificOutputsFromDataset(dataset, examplesPredict)
    #print(Orange.evaluation.scoring.confusion_matrix(realValuesPredict, predictions))
    #print(ut.print_results(realValuesPredict,predictions))

    '''
        APLICACAO DO KMEANS ALGORITHM
    '''
    print("\nLoading K-Means\n")
    #IDENTIFICACAO DO MELHOR VALOR DE K (CLUSTERS), TENDO EM CONTA UMA GAMA DE CLUSTERS E O DATASET EM ANALISE
    transposeDataset = ut.applyTranspostMatrix(dataset)
    #bestValueofK = ut.getBestValueOfK(dataset)
    transposeDataset = ut.applyMinMaxScaler(transposeDataset)
    #print(transposeDataset.getDataset().X)
    kMeansObject = ut.applyClustering(60, transposeDataset)
    #print(kMeansObject.getLearner().cluster_centers_.shape)
    #print(kMeansObject.getLearner().labels_.shape)
    #print(numpy.argwhere(kMeansObject.getLearner().labels_ == 3))
    arrayBestFeatures = ut.getBestValuesForCluster(2,kMeansObject,transposeDataset)
    #ARRAY WITH RELEVANT FEATURES --> ARRAY WITH 0'S OR 1'S --> 1'S RELEVANT FEATURES
    myArray = ut.createBinaryNumpyArrayWithReducedFeatures(arrayBestFeatures, dataset)

    #GET DATASET WITH FEATURE REDUCTION --> AFTER APPLY KMEANS ALGORITHM
    reducedDataset = ut.createCloneOfReducedDataset(dataset, myArray)

    # listSamplesPredict = ut.getSpecificSamples(reducedDataset, examplesPredict)
    # predictions = learner.fit(reducedDataset.getDataset().X[examplesTraining],reducedDataset.getDataset().Y[examplesTraining]).predict(listSamplesPredict)
    # realValuesPredict = ut.getSpecificOutputsFromDataset(reducedDataset, examplesPredict)
    # print(Orange.evaluation.scoring.confusion_matrix(realValuesPredict, predictions))
    # print(ut.print_results(realValuesPredict,predictions))

    #APLICACAO DE CROSS VALIDATION
    reducedDataset = ut.applyMinMaxScaler(reducedDataset)
    print("Score e Previsoes KMeans - Cross Validation")
    kFold = KFold(n_splits=4, shuffle=True)
    result = cross_val_score(learner, reducedDataset.getDataset().X, reducedDataset.getDataset().Y, cv=kFold, scoring='r2')
    print("Media Score:\t",result.mean())
    predictions = cross_val_predict(learner, reducedDataset.getDataset().X, reducedDataset.getDataset().Y, cv=3)
    print("Prediction:\t",predictions)
    print("\nSVM")
    learner.fit(reducedDataset.getDataset().X[examplesTraining], reducedDataset.getDataset().Y[examplesTraining])
    listSamplesPredict = ut.getSpecificSamples(reducedDataset, examplesPredict)
    predictions = learner.predict(listSamplesPredict)
    realValuesPredict = ut.getSpecificOutputsFromDataset(reducedDataset, examplesPredict)
    print(Orange.evaluation.scoring.confusion_matrix(realValuesPredict, predictions))
    print(ut.print_results(realValuesPredict,predictions))

    '''
        APLICACAO DO BINARY PSO
    '''
    print("\nLoading BPSO\n")
    # '''
    #     ABERTURA DA APLICACAO
    #     app = QApplication(sys.argv)
    #     window = MainWindow.MainWindow(dataset.getDataset())
    # '''

    #DEFINICAO DO ALGORITMO PSO
    n_particles = 50
    psoArgs = {UtilsPSO.UtilsPSO.INERCIA: 0.9, UtilsPSO.UtilsPSO.C1 : 1.4, UtilsPSO.UtilsPSO.C2 : 1.4, UtilsPSO.UtilsPSO.ALPHA : 0.80, UtilsPSO.UtilsPSO.NEIGHBORS : n_particles, 'p': 2} #p não é relevante, visto que todas as particulas se veem umas as outras, o p representa a distancia entre cada uma das particulas

    psoAlgorithm = factory.getUtil(ut.PSO).getPso(**psoArgs)
    optionsPySwarms = {'c1' : psoAlgorithm.getC1(), 'c2' : psoAlgorithm.getC2(), 'w' : psoAlgorithm.getInercia(), 'k' : psoArgs.get(UtilsPSO.UtilsPSO.NEIGHBORS), 'p' : psoArgs.get('p')}

    #dimensionsOfProblem = dataset.getDataset().X.shape[1] #FEATURES DO DATASET
    dimensionsOfProblem = reducedDataset.getDataset().X.shape[1]
    initPos = ut.createArrayInitialPos(n_particles,dimensionsOfProblem,dimensionsOfProblem-4) #COLOCANDO UM NUMERO BAIXO NO INIT_POS, PERCEBEMOS QUE AQUI A ACCURACY JA VAI ALTERANDO, POIS COMO EXISTEM POUCAS AMOSTRAS, A ACCURACY REVELA-SE QUASE SEMPRE IGUAL, MESMO TREINANDO DUAS AMOSTRAS COM ATRIBUTOS SEMELHANTES E OUTPUTS DISTINTOS
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensionsOfProblem, options=optionsPySwarms)
    bestCost, bestPos = optimizer.optimize(psoAlgorithm.aplicarFuncaoObjetivoTodasParticulas, 200, dataset= reducedDataset, classifier=classificador, alpha=psoAlgorithm.getAlpha())

    #CONTAGEM DE QUANTAS FEATURES SAO RELEVANTES
    bestPos = ut.listToNumpy(bestPos)
    newFeatures = numpy.count_nonzero(bestPos)
    #print(newFeatures)

    #CRIACAO DA COPIA
    deepCopy = ut.createCloneOfReducedDataset(reducedDataset,bestPos)
    print(deepCopy.getDataset().X.shape)

    '''
        PREVISAO FINAL DAS 4 AMOSTRAS QUE NAO FORAM CONSIDERADAS NO TREINO DA APLICACAO DOS ALGORITMOS ANTERIORES
    '''

    #TREINO E PREVISAO, APENAS COM AS FEATURES SELECCIONADAS
    deepCopy = ut.applyMinMaxScaler(deepCopy)
    kFold = KFold(n_splits=4, shuffle=True)
    result = cross_val_score(learner, deepCopy.getDataset().X, deepCopy.getDataset().Y, cv=kFold, scoring='r2')
    print("Media Score:\t",result.mean())
    predictions = cross_val_predict(learner, deepCopy.getDataset().X, deepCopy.getDataset().Y, cv=3)
    print("Real:\t",deepCopy.getDataset().Y)
    print("Prediction:\t",predictions)
    print(Orange.evaluation.scoring.confusion_matrix(deepCopy.getDataset().Y, predictions))
    print(ut.print_results(deepCopy.getDataset().Y,predictions))
    print("\nFinal Results: SVM\n")
    listSamplesPredict = ut.getSpecificSamples(deepCopy, examplesPredict)
    predictionsAfterFeatureSelection = learner.fit(deepCopy.getDataset().X[examplesTraining],deepCopy.getDataset().Y[examplesTraining]).predict(listSamplesPredict)
    realValuesPredict = ut.getSpecificOutputsFromDataset(deepCopy, examplesPredict)
    print(Orange.evaluation.scoring.confusion_matrix(realValuesPredict, predictionsAfterFeatureSelection))
    print(ut.print_results(realValuesPredict,predictionsAfterFeatureSelection))

    # '''
    #     FECHO DA APLICACAO
    #     sys.exit(app.exec())
    # '''

if __name__== "__main__":
    main()
