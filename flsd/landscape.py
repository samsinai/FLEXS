import abc

class Landscape(abc.ABC):

    @abc.abstractmethod
    def get_fitness(self, sequences):
        pass