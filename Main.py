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

def main():

    '''
        Assim era outra forma de fazer as coisas, mas o dataset nao esta a vir muito bem
        dataset = GEOparse.get_GEO(geo="GDS360", destdir="./")
        ds = ut.pandas_to_orange(dataset.table)
        print(ds.X.shape)
    '''

    data = Orange.data.Table("./datasetExplore")
    dataset = UtilsFactory.UtilsFactory.getUtil(ut.DATASET).getDataset(data)
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
    classificador = UtilsFactory.UtilsFactory.getUtil(ut.CLASSIFIER).getClassifier(gamma=0.1, vizinhos=5) #CRIACAO DO CLASSIFICADOR

    #DEFINICAO DO ALGORITMO PSO
    n_particles = 50
    psoArgs = {UtilsPSO.UtilsPSO.alpha: 0.9, UtilsPSO.UtilsPSO.C1 : ut.generateRandomValue(0,1), UtilsPSO.UtilsPSO.C2 : ut.generateRandomValue(0,1), UtilsPSO.UtilsPSO.ALPHA : ut.generateRandomValue(0,1), UtilsPSO.UtilsPSO.NEIGHBORS : n_particles, 'p': 2} #p não é relevante, visto que todas as particulas se veem umas as outras, o p representa a distancia entre cada uma das particulas

    psoAlgorithm = UtilsFactory.UtilsFactory.getUtil(ut.PSO).getPso(**psoArgs)
    optionsPySwarms = {'c1' : psoAlgorithm.getC1(), 'c2' : psoAlgorithm.getC2(), 'w' : psoAlgorithm.getInercia(), 'k' : psoArgs.get(UtilsPSO.UtilsPSO.NEIGHBORS), 'p' : psoArgs.get('p')}

    dimensionsOfProblem = dataset.getDataset().X.shape[1] #FEATURES DO DATASET
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensionsOfProblem, options=optionsPySwarms)

    cost, pos = optimizer.optimize(psoAlgorithm.aplicarFuncaoObjetivoTodasParticulas, dataset= dataset, classifier=classificador, alpha=psoAlgorithm.getAlpha(), iters=100)


if __name__== "__main__":
    main()
