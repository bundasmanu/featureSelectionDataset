from abc import ABC, abstractmethod

#BOA EXPLICACAO DO SIGNIFICADO DE CROSS VALIDATION

class AbstractLearner(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getLearner(self):
        pass