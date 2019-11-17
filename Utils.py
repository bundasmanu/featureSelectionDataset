import numpy as np
import pandas as pd
import Orange
import csv
from io import StringIO
from collections import OrderedDict
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import UtilsDataset
import random
import UtilsFactory
import Orange.evaluation.scoring
import sklearn.metrics
import AbstractLearner, KMeansLearner
from sklearn.metrics import silhouette_score
import scipy.spatial.distance as sdist
from nltk import flatten
from sklearn.preprocessing import MinMaxScaler

CLASSIFIER = 'classifier'
DATASET = 'dataset'
PSO = 'pso'
MAINWINDOWTITLE = "PSO FEATURE SELECTION"

rangeClusterValues = [6, 12, 18]

'''

Funcoes de Conversao de pandas.dataframe para Orange.data.Table, nao foi necessario o seu uso

'''
def pandas_to_orange(df):
    domain, attributes, metas = construct_domain(df)
    orange_table = Orange.data.Table.from_numpy(domain = domain, X = df[attributes].values, Y = None, metas = df[metas].values, W = None)
    return orange_table

def construct_domain(df):
    columns = OrderedDict(df.dtypes)
    attributes = OrderedDict()
    metas = OrderedDict()
    for name, dtype in columns.items():

        if issubclass(dtype.type, np.number):
            if len(df[name].unique()) >= 13 or issubclass(dtype.type, np.inexact) or (df[name].max() > len(df[name].unique())):
                attributes[name] = Orange.data.ContinuousVariable(name)
            else:
                df[name] = df[name].astype(str)
                attributes[name] = Orange.data.DiscreteVariable(name, values = sorted(df[name].unique().tolist()))
        else:
            metas[name] = Orange.data.StringVariable(name)

    domain = Orange.data.Domain(attributes = attributes.values(), metas = metas.values())

    return domain, list(attributes.keys()), list(metas.keys())

'''

Funcoes que manipulam o conteudo do dataset recebido inicialmente no ficheiro tab,
o seu conteudo encontra-se mal representado por exemplo:
- os targets nao tem conteudo: necessario definir o array de targets
- os dados "array X", conteudo as classes (targets) e metadata: é necessario eliminar estes dados (cada coluna do array X)
'''

def transformMatrixDatasetInCorrectFormat(dataset : UtilsDataset.UtilsDataset):

    #TRANSFORMACAO INICIAL DAS COLUNAS#
    defineColumnsTable(dataset.getDataset())

    #TIRAR COLUNAS A MAIS DOS DADOS--> ARRAY X, OU SEJA, LEN(X)-FEATURES = ULTIMOS DADOS A MAIS
    deleteColumnsMoreOverData_X(dataset.getDataset())

    return dataset

def defineColumnsTable(table : Orange.data.Table):

    #CHAMADA FUNCAO getIndexOutput --> POSICAO DO ATRIBUTO CLASSE --> TARGETS DO DATASET
    indexOfClass = getIndexOutput(table)

    #AGORA BASTA POPULAR O ARRAY Y --> COM OS TARGETS REFERENTES A CADA UMA DAS LINHAS DE DADOS (X)--> NA COLUNA REFERENTE À CLASSE (INDICE RETORNADO)
    allOutputs = np.array([output for output in table.X[:,indexOfClass]]) #--> 24 valores, visto que sao 24 linhas neste exemplo

    table.Y = allOutputs #--> COMO O TABLE.Y É UM NUMPY ARRAY, TIVE DE CONVERTER O ALLOUTPUTS PARA NUMPY ARRAY

    return table

def getIndexOutput(table : Orange.data.Table):
    return table.domain.index("class")

def deleteColumnsMoreOverData_X(table : Orange.data.Table):

    #GET INDEX OF ATTRIBUTE CLASS ON DATA
    classIndex = getIndexOutput(table)

    #GET TOTAL COLUMNS DATA
    columns = table.X.shape[1]

    #APAGAR COLUNAS QUE VAO DESDE A CLASS INDEX ATE AO INDEX COLUMNS --> REFORMULAR MATRIZ X
    for j in range(table.X.shape[1]-1, classIndex-1, -1): #TEM DE ESTAR AO CONTRARTO (DO ULTIMO PARA O INDEX DA CLASSE) PORQUE COMO APAGO, FICAM MENOS COLUNAS, E SE COMECASSE PELA ORDEM HABITUAL, GERAVA INDEX OUT OF BOUNDS, PORQUE ACEDI A COLUNAS QUE JA FORAM APAGADAS, COMECANDO AO CONTRARIO NAO HÁ PROBLEMA
        table.X = np.delete(table.X, j, axis=1) #AXIS = 1 --> REPRESENTA O EIXO DAS COLUNAS

    return table

'''
    GENERATE RANDOM VALUE BETWEEN TWO VALUES --> 2 DECIMAL CASES
'''

def generateRandomValue(minLimit,maxLimit):
    return round(random.uniform(minLimit,maxLimit),2)

'''
    TRANSFORM LIST INTO NUMPY ARRAY
'''
def listToNumpy(list):
    return np.array(list)

'''
    CRIACAO DE UM DATASET QUE É UMA COPIA DO DATASET INICIAL, MAS COM UM NUMERO DE FEATURES MAIS COMPACTA, DEPOIS DE APLICADO O ALGORITMO, OS TARGETS MANTEM-SE
'''
def createCloneOfReducedDataset(dataset : UtilsDataset.UtilsDataset, bestPos):

    try:

        if not isinstance(dataset, UtilsDataset.UtilsDataset):
            raise TypeError

        #CRIACAO DE UM NOVO OBJETO UTILS DATASET E DEPOIS EDITAR O DATASET
        copyOfDataset = dataset.deepCopy()

        #RECRIACAO DO ARRAY X TENDO EM CONTA A REFORMULACAO DE DADOS --> NECESSITO DO BESTPOS (ARRAY) RETORNADO PELO PSO--> MELHOR POSICAO ENCONTRADA
        copyOfDataset.getDataset().X = np.delete(copyOfDataset.getDataset().X, np.argwhere(bestPos==0), axis=1)

        return copyOfDataset

    except:
        print('Catched error')
        return None

'''
    FUNCAO QUE IMPRIME OS RESULTADOS DA PREVISAO
'''
def print_results(real, predictions):

    '''
        I need to pass average and labels arguments to metrics, because i'm just trying to predict x samples, and not everything
        Link: https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi/47285662
    '''

    print("Real:\t",real)
    print("Previsoes:\t",predictions)
    loss = sklearn.metrics.f1_score(real,predictions, average='weighted', labels=np.unique(predictions))
    accuracy = sklearn.metrics.accuracy_score(real,predictions)
    precision = sklearn.metrics.precision_score(real,predictions, average='weighted', labels=np.unique(predictions))
    recall = sklearn.metrics.recall_score(real,predictions, average='weighted', labels=np.unique(predictions))
    print('f1_score=', loss)
    print('accuracy=', accuracy)
    print ('precision=', precision)
    print ('recall=', recall)

'''
    FUNCOES QUE APLICAM A IDEIA DO ALGORITMO DE SELECCAO DE FEATURES, RECORRENDO AO PSO
'''

def createEmptyNumpyArray(nParticles, nDimensions):

    '''

    :param nParticles: nº de particulas
    :param nDimensions: nº de dimensoes do problema
    :return: numpy Array com shape de (nParticles, nDimensions), com todos os valores a 0
    '''

    emptyArray = np.zeros(shape=(nParticles,nDimensions))

    return emptyArray

def generateListRandomValues(min, max, numberOfValues):

    '''
    THIS METHOD AVOIDS REPEATED NUMBERS

    :param min: min value on list --> em principio 0
    :param max: max value on list --> number of Particles
    :param numberOfValues: numberValues to sort --> por exemplo quero apenas 100 features
    :return: array with numberOfValues values sorted
    '''

    listAllValues = list(range(min,max))
    random.shuffle(listAllValues)

    myList = list() #CRIACAO DE UMA LISTA VAZIA

    for i in range(numberOfValues):
        myList.append(listAllValues.pop()) #VAI BUSCAR UM VALOR AO ARRAY SORTEADO

    return myList

def selectRelevantFeaturesByParticle(indexParticle, listFeatures, listParticlesDimensions):

    '''

    :param indexParticle: indice da particula
    :param listFeatures: features relevantes sorteadas (para colcar na particula com valor 1)
    :param listParticlesDimensions: lista com as posicoes das particulas
    :return: lista atualizada--> valores das features de apenas uma particula
    '''

    for j in listFeatures:
        listParticlesDimensions[indexParticle][j] = 1

    return listParticlesDimensions

def createArrayInitialPos(nParticles, nDimensions, nRelevantFeatures):

    '''

    :param n_particles: numero de particulas a utilizar no algoritmo
    :param nDimensions: dimensoes do problema (nº de features)
    :param nRelevantFeatures: numero de features que devem estar a 1 (sao relevantes)
    :return: numpy array, com shape de (nParticles, nDimensions), preenchido com valores a 1 = nRelevantFeatures --> aleatoriamente
    '''

    emptyArray = createEmptyNumpyArray(nParticles, nDimensions)

    for i in range(nParticles):
        listRandomValues = generateListRandomValues(0, nDimensions, nRelevantFeatures)
        emptyArray = selectRelevantFeaturesByParticle(i,listRandomValues,emptyArray)

    return emptyArray

'''
    CLUSTERS
'''

def applyTranspostMatrix(dataset : UtilsDataset.UtilsDataset):

    '''

    :param dataset: matriz do dataset a fazer transposicao
    :return: deep copy da matriz transposta
    '''

    transposeMatrix = dataset.deepCopy()

    transposeMatrix.getDataset().X = np.transpose(transposeMatrix.getDataset().X)

    return transposeMatrix

def getBestValueOfK(dataset : UtilsDataset.UtilsDataset):

    '''
    Fontes : https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
            https://stats.stackexchange.com/questions/10540/how-to-interpret-mean-of-silhouette-plot
    :param listRangeValuesK: lista com possiveis valores de K
    :return: valor de k, que apresenta melhor score, recorrendo ao silhouette
    '''
    bestKValue = rangeClusterValues[0]
    silhouetteAvg = 0.0
    for cluster in rangeClusterValues:
        clusterInEvaluation = KMeansLearner.KMeansLearner(cluster)

        clusterPredictions = clusterInEvaluation.getLearner().fit_predict(dataset.getDataset().X)

        silAvg = silhouette_score(dataset.getDataset().X, clusterPredictions)

        if silAvg > silhouetteAvg : #MAIOR INTERDEPENDENCIA ENTRE CLUSTERS
            silhouetteAvg = silAvg
            bestKValue = cluster

    return bestKValue

def applyClustering(clusterNumber, dataset : UtilsDataset.UtilsDataset):
    '''

    :param dataset: dataset a treinar
    :return: retorno do objeto KMeans utilizado no treino
    '''

    kmeans = KMeansLearner.KMeansLearner(clusterNumber) #CRIACAO DO OBJETO
    kmeans.getLearner().fit_predict(dataset.getDataset().X) #TREINO

    return kmeans

def findLabelWithSpecificDistance(distance, kMeans : KMeansLearner.KMeansLearner, dataset: UtilsDataset.UtilsDataset):
    '''
    :param distance : distancia
    :param kMeans: objeto kmeans
    :param dataset: dataset
    :return: retorno feature (posicao) relativa à distancia passada por argumento
    '''

    for i in range(len(kMeans.getLearner().cluster_centers_)):
        for j in range(len(kMeans.getLearner().labels_)):
            if i == kMeans.getLearner().labels_[j]:
                if (np.linalg.norm(dataset.getDataset().X[j] - kMeans.getLearner().cluster_centers_[i]) == distance):
                    return j


def getBestValuesForCluster(manyValues, kMeans : KMeansLearner.KMeansLearner, dataset: UtilsDataset.UtilsDataset):
    '''

    :param manyValues: quantos valores pretendo por cluster
    :param kMeans: objeto KMeans
    :return: lista com os atributos mais relevantes, de cada cluster, matriz[linhas, colunas]--> linhas = features de cada cluster
    '''
    #Fonte: https://stackoverflow.com/questions/51309526/kmeans-euclidean-distance-to-each-centroid-avoid-splitting-features-from-rest-of

    controlNumberValues = 0
    arrayBestFeaturesPerCluster = [[None]*manyValues for i in range(60)]
    arrayPositionsPerCluster = [[None]*manyValues for i in range(60)]

    for i in range(len(kMeans.getLearner().cluster_centers_)):
        controlNumberValues = 0
        for j in range(len(kMeans.getLearner().labels_)):
            if i == kMeans.getLearner().labels_[j]:
                distance = np.linalg.norm(dataset.getDataset().X[j]- kMeans.getLearner().cluster_centers_[i])
                if (len(arrayBestFeaturesPerCluster[i]) - arrayBestFeaturesPerCluster[i].count(None))< manyValues: #SE AINDA TEM ESPACO COLOCA LA A DISTANCIA
                    arrayBestFeaturesPerCluster[i][controlNumberValues] = distance
                    arrayPositionsPerCluster[i][controlNumberValues] = j
                else:
                    arrayBestFeaturesPerCluster[i].sort() #COLOCO POR ORDEM OS VALORES, OU SEJA O MELHOR NO INICIO E O PIOR NO FIM
                    if arrayBestFeaturesPerCluster[i][controlNumberValues] > distance: #SE O NOVO VALOR FOR MENOR, QUE O PIOR, COLOCO-O LÁ
                        indexWorstPosition = findLabelWithSpecificDistance(arrayBestFeaturesPerCluster[i][controlNumberValues], kMeans,dataset) #OBTENCAO DO INDICE COM A PIOR POSICAO
                        indexToRemove = arrayPositionsPerCluster[i].index(indexWorstPosition) #OBTENCAO DO INDICE ONDE ESTA A POSICAO COM PIOR DISTANCIA
                        arrayPositionsPerCluster[i][indexToRemove] = j
                        arrayBestFeaturesPerCluster[i][controlNumberValues] = distance #ATUALIZO A DISTANCIA
                if controlNumberValues < manyValues-1: #APENAS ITERO A VARIAVEL QUANDO NAO TENHO AINDA OS ELEMENTOS QUE PRETENDO, CASO CONTRARIO FICA SEMPRE NA ULTIMA POSICAO
                    controlNumberValues+=1

    return arrayPositionsPerCluster

'''
    TRANSFORM DATASET IN REDUCED DATASET --> USING CLONE FUNCTION--> IMPORT FOR CREATE CLONE OF NEW'S DATASET
'''

def createBinaryNumpyArrayWithReducedFeatures(relevantFeatures, dataset: UtilsDataset.UtilsDataset):
    '''

    :param relevantFeatures: array with relevant features
    :return: numpy binary array with 0 or 1 --> 1 relevant features
    '''

    oneDArray = flatten(relevantFeatures) #PASSAGEM DE 2D ARRAY PARA 1D ARRAY

    oneDArray = [i for i in oneDArray if i is not None]#ELIMINACAO DE POSSIVEIS NONE'S QUE POSSAM EXISTIR
    #print(oneDArray)
    emptyArray = np.zeros(dataset.getDataset().X.shape[1]) #PASSO O DATASET OFICIAL E NAO O TRANSPOSTO

    for i in oneDArray:
        emptyArray[i] = 1

    return emptyArray

'''
    APPLY SCALER TO FEATURES OF DATASET, USING MINMAXSCALER
'''

def applyMinMaxScaler(dataset : UtilsDataset.UtilsDataset):

    scaler = MinMaxScaler()

    scaler.fit(dataset.getDataset().X) #COMPUTACAO DOS VALORES DE MIN E MAX A UTILIZAR NA COMPUTACAO

    dataset.getDataset().X = scaler.transform(dataset.getDataset().X) #APLICACAO DA ESCALA AS FEATURES DO DATASET, DE ACORDO COM O RANGE ESTABELECIDO NO FIT
    #O SCALER TRANSFORM RETORNA O DATASET ALTERADO --> CONVEM FAZER COPIAS, PARA MANTER OS DADOS ORIGINAIS SEGUROS

    return dataset

def getSpecificSamples(dataset : UtilsDataset.UtilsDataset, indexOfSamples):
    '''

    :param dataset: dataset of problem
    :param indexOfSamples: indexes that i want from my dataset
    :return: numpy array with specific data (specific samples)
    '''

    listSpecificSamples = np.array([dataset.getDataset().X[i] for i in indexOfSamples])

    return listSpecificSamples

def getSpecificOutputsFromDataset(dataset : UtilsDataset.UtilsDataset, indexOfOutputs):
    '''

    :param dataset: dataset of problem
    :param indexOfOutputs: list of indexes of outputs that i want
    :return: numpy array with values from specific indexes of outputs
    '''

    listSpecificOutputs = np.array([dataset.getDataset().Y[i] for i in indexOfOutputs])

    return listSpecificOutputs