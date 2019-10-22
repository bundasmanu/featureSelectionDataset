import Orange

class UtilsDataset():

    def __init__(self, myDataset : Orange.data.Table):
        self.dataset = myDataset

    def getDataset(self):
        return self.dataset

    def setNewDataset(self, newDataset : Orange.data.Table):
        self.dataset = newDataset