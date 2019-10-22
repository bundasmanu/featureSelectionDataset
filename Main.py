import pyswarms as ps
import Orange
import GEOparse
import Utils as ut
from Orange.data.pandas_compat import table_from_frame
from Bio import Geo
import UtilsDataset as udt
import UtilsFactory

def main():

    '''
        Assim era outra forma de fazer as coisas, mas o dataset nao esta a vir muito bem
        dataset = GEOparse.get_GEO(geo="GDS360", destdir="./")
        ds = ut.pandas_to_orange(dataset.table)
        print(ds.X.shape)
    '''

    data = Orange.data.Table("./datasetExplore")
    dataset = udt.UtilsDataset(data)
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

if __name__== "__main__":
    main()
