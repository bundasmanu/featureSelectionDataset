import Orange
import copy

class UtilsDataset():

    def __init__(self, myDataset : Orange.data.Table):
        self.dataset = myDataset

    def deepCopy(self):
        return copy.deepcopy(self)


    def getDataset(self):
        return self.dataset

    def setNewDataset(self, newDataset : Orange.data.Table):
        self.dataset = newDataset