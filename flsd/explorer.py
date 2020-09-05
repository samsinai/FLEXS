import abc

class Explorer(abc.ABC):

    def set_model(self, model):
        pass

    def propose_samples(self):
        pass

    def measure_proposals(self, proposals):
        pass
