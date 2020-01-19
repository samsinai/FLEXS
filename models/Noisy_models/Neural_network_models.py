
import editdistance
import numpy as np
from meta.model import Model
import random
from utils.sequence_utils import translate_string_to_one_hot
from sklearn.metrics import explained_variance_score, r2_score
import keras

class NN_model(Model):

    def __init__(self, ground_truth_oracle, nn_architecture, cache=True, batch_update=False, landscape_id=-1, start_id=-1):
        self.measured_sequences = {} # save the measured sequences for the model
        self.model_sequences = {} # cache the sequences for later queries
        self.cost = 0
        self.evals = 0
        self.r2 = 0
        self.cache = cache
        self.landscape_id = landscape_id
        self.start_id = start_id
        self.oracle = ground_truth_oracle
        self.batch_update = batch_update
        self.neuralmodel = nn_architecture.get_model()
        self.batch_size = nn_architecture.batch_size
        self.validation_split = nn_architecture.validation_split
        self.epochs = nn_architecture.epochs
        self.alphabet = nn_architecture.alphabet
        self.nn_architeture = nn_architecture
        self.one_hot_sequences = {}
        self.model_type =f'nn_arch={nn_architecture.architecture_name}'

    def reset(self,sequences= None):
        self.model_sequences = {}
        self.measured_sequences = {}
        self.cost = 0
        self.neuralmodel=keras.models.clone_model(self.neuralmodel)
        self.neuralmodel.compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])
        if sequences:
            self.update_model(sequences)
    # def reset(self, nn_architeture):
    #     self.neuralmodel = nn_architeture.get_model()
    #     self.model_sequences={}
    #     self.measured_sequences={}
    #     self.cost=0

    def bootstrap(self,wt,alphabet):
        sequences=[wt]
        self.wt=wt
        self.alphabet=alphabet
        for i in range(len(wt)):
            tmp=list(wt)
            for j in range(len(alphabet)):
                tmp[i]=alphabet[j]
                sequences.append("".join(tmp))
        self.measure_true_landscape(sequences)
        self.one_hot_sequences={sequence:(translate_string_to_one_hot(sequence,self.alphabet),self.measured_sequences[sequence]) for sequence in sequences} #removed flatten for nn
        self.update_model(sequences)



    def update_model(self,sequences):
        X=[]
        Y=[]
        self.measure_true_landscape(sequences)
        for sequence in sequences:
            if sequence not in self.one_hot_sequences:# or self.batch_update:
                x=translate_string_to_one_hot(sequence,self.alphabet)#.flatten()
                y=self.measured_sequences[sequence]
                self.one_hot_sequences[sequence]=(x,y)
                X.append(x)
                Y.append(y)
            else:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)
        X=np.array(X)
        Y=np.array(Y)

        try:
            y_pred=self.neuralmodel.predict(X)
            self.r2=r2_score(Y,y_pred)

        except:
            pass

        self.neuralmodel.fit(X,Y,epochs=self.epochs,validation_split=self.validation_split,batch_size=self.batch_size,verbose=0)
        if not self.batch_update:
            self.retrain_model()

    def retrain_model(self):
        X,Y=[],[]
        random_sequences=random.sample(self.one_hot_sequences.keys(),min(len(self.one_hot_sequences.keys()),self.batch_size*5))
        for sequence in random_sequences:
                x,y=self.one_hot_sequences[sequence]
                X.append(x)
                Y.append(y)

        X=np.array(X)
        Y=np.array(Y)

        self.neuralmodel.fit(X,Y,epochs=self.epochs,validation_split=self.validation_split,batch_size=self.batch_size,verbose=0)


    def _fitness_function(self,sequence):
        try:
            x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]
        except:
            print (sequence)
        return max(min(200, self.neuralmodel.predict(x)[0][0]),-200)
    
    def measure_true_landscape(self,sequences):
        for sequence in sequences:
            if sequence not in self.measured_sequences:
                    self.cost += 1
                    self.measured_sequences[sequence]=self.oracle.get_fitness(sequence)

        self.model_sequences = {} #empty cache


    def get_fitness(self,sequence):

        if sequence in self.measured_sequences: 
            return self.measured_sequences[sequence]
        elif sequence in self.model_sequences and self.cache: #caching model answer to save computation
            return self.model_sequences[sequence]

        else:
            self.model_sequences[sequence] = self._fitness_function(sequence)
            self.evals += 1
            return self.model_sequences[sequence]   

# class Ensemble_best_model(NN_model):

#         def assign_models(self,list_of_models_dict):
#             self.list_of_models_dict=list_of_models_dict
#             #[{'model:<model_object','epochs':epochs,'batch_size':batch_size},{}]
#             self.model_performances=[0 for i in range(len(self.list_of_models_dict))]
#             self.best_model=self.list_of_models_dict[0]['model']

#         def reset(self):
#             self.measured_sequences={}
#             self.cost=0
#             self.model_performances=[0 for i in range(len(self.list_of_models_dict))]

#             for model_dict in self.list_of_models_dict:
#                 model_dict['model']=keras.models.clone_model(model_dict['model'])
#                 model_dict['model'].compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])
       
#         def bootstrap(self,wt,alphabet):
#             sequences=[wt]
#             self.wt=wt
#             self.alphabet=alphabet
#             for i in range(len(wt)):
#                 tmp=list(wt)
#                 for j in range(len(alphabet)):
#                     tmp[i]=alphabet[j]
#                     sequences.append("".join(tmp))
#             self.measure_true_landscape(sequences)
#             self.one_hot_sequences={sequence:(translate_string_to_one_hot(sequence,self.alphabet),self.measured_sequences[sequence]) for sequence in sequences} #removed flatten for nn
#             self.update_model(sequences)


#         def update_model(self,sequences, bootstrap=True):
#             X=[]
#             Y=[]
#             for sequence in sequences:
#                 if sequence not in self.one_hot_sequences:# or self.batch_update:
#                     x=translate_string_to_one_hot(sequence,self.alphabet)#.flatten()
#                     y=self.measured_sequences[sequence]
#                     self.one_hot_sequences[sequence]=(x,y)
#                     X.append(x)
#                     Y.append(y)
#                 else:
#                     x,y=self.one_hot_sequences[sequence]
#                     X.append(x)
#                     Y.append(y)
#             X=np.array(X)
#             Y=np.array(Y)

#             for i,model_dict in enumerate(self.list_of_models_dict):
#                 try:
#                     y_pred=model_dict['model'].predict(X)
#                     self.model_performances[i]=r2_score(Y,y_pred)

#                 except:
#                     pass
#                 indices=[]
#                 for k in range(int(len(X)/2)):
#                     indices.append(random.randint(0,len(X)-1))
#                 model_dict['model'].fit(np.take(X,indices,axis=0),np.take(Y,indices,axis=0),epochs=model_dict['epochs'],validation_split=0,batch_size=model_dict['batch_size'],verbose=0)
#             best_model_index=np.argmax(self.model_performances)
#             self.best_model=self.list_of_models_dict[best_model_index]['model']


#         def _fitness_function(self,sequence):
#             x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]

#             return max(min(200, self.best_model.predict(x)[0][0]),-200)

# class Ensemble_mean_dist(Ensemble_best_model):

#         def get_uncertainty(self,sequence):
#             x=np.array([translate_string_to_one_hot(sequence,self.alphabet)])#.flatten()]

#             x_predicts=[m["model"].predict(x)[0][0] for m in self.list_of_models_dict]

#             return np.mean(x_predicts), np.std(x_predicts), x_predicts

#         def _fitness_function(self,sequence):
#             x_mean,x_std, all_x=self.get_uncertainty(sequence)

#             return max(min(200, x_mean),-200)



