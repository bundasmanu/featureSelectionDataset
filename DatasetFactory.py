import UtilsDataset
import Orange

class DatasetFactory:

    def getDataset(self, myDataset : Orange.data.Table):
        return UtilsDataset.UtilsDataset(myDataset)