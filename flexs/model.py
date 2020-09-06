import abc

import flexs


class Model(flexs.Landscape, abc.ABC):
    """Base structure for all models"""

    """This function is called whenever you would want your model to update itself based on the set of sequnecs it has measurements for"""

    @abc.abstractmethod
    def train(self, data):
        pass
