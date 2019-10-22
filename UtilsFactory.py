import Utils as ut
import DatasetFactory, PSOFactory, ClassifierFactory


class UtilsFactory():

    def __init__(self):
        pass

    def getUtil(self, name):

        if name == ut.DATASET:
            return DatasetFactory.DatasetFactory()
        elif name == ut.PSO:
            return PSOFactory.PSOFactory()
        elif name == ut.CLASSIFIER:
            return ClassifierFactory.ClassifierFactory()
        else:
            return None
