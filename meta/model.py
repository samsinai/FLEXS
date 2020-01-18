
class Ground_truth_oracle():
    def __init__(self):
        pass

    """ This is your true model, given a sequence, it should generate its fitness. Can be a pretrained model on data.
     Should be treated as a private function."""
    def _fitness_function(self,sequence):    
        pass

    """This is a public wrapper on fitness function, it allows faster lookup, as well as, if you want, overriding your model with actual data"""
    def get_fitness(self,sequence):
        pass



class Model(Ground_truth_oracle):
    """Base structure for all models"""
    def __init__(self):
        pass

    """This function is called whenever you would want you model to update itself based on the set of sequnecs it has measurements for"""
    def update_model(self,sequences):
        pass





    

    
