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

CLASSIFIER = 'classifier'
DATASET = 'dataset'
PSO = 'pso'

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

    loss = sklearn.metrics.hamming_loss(real,predictions)
    accuracy = sklearn.metrics.accuracy_score(real,predictions)
    precision = sklearn.metrics.precision_score(real,predictions)
    recall = sklearn.metrics.recall_score(real,predictions)
    print('loss=', loss)
    print('accuracy=', accuracy)
    print ('precision=', precision)
    print ('recall=', recall)