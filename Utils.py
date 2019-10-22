import numpy as np
import pandas as pd
import Orange
import csv
from io import StringIO
from collections import OrderedDict
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

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

