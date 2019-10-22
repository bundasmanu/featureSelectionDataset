import UtilsPSO

class PSOFactory:

    def getPso(self, **attributes):
        return UtilsPSO.UtilsPSO(attributes)