import sys
from sklearn.linear_model import LinearRegression,Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D

class Architecture():
    def __init__(self, seq_len, batch_size=10, validation_split=0.1, epochs=20, alphabet="UCGA"):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.architecture_name='LNN'

    @property
    def alphabet_len(self):
        return len(self.alphabet)

    def get_model(self):
        raise NotImplementedError( "You need to define an Architecture")


class Linear(Architecture):

    def get_model(self):

        lin_model=Sequential()
        lin_model.add(Flatten())
        lin_model.add(Dense(1,input_shape=(self.seq_len* self.alphabet_len,),use_bias=False))
        lin_model.add(Activation('linear')) 
        lin_model.compile(loss='mean_squared_error', optimizer="adam",metrics=['mse'])  
        return lin_model

class NLNN(Architecture):
    """Global epistasis model"""
    def __init__(self, seq_len, batch_size=10, validation_split=0.1, epochs=20, alphabet="UCGA", hidden_dims=50):
        super(NLNN, self).__init__(seq_len, batch_size, validation_split, epochs, alphabet)
        self.hidden_dims = hidden_dims
        self.architecture_name=f'NLNN_hd{self.hidden_dims}'


    def get_model(self):

        non_lin_model=Sequential()
        non_lin_model.add(Flatten())
        non_lin_model.add(Dense(1,input_shape=(self.seq_len* self.alphabet_len,),use_bias=False))
        non_lin_model.add(Activation('linear'))
        non_lin_model.add(Dense(self.hidden_dims))
        non_lin_model.add(Activation("relu"))
        non_lin_model.add(Dense(self.hidden_dims))
        non_lin_model.add(Activation("relu"))
        non_lin_model.add(Dense(1))
        non_lin_model.add(Activation("linear"))
        non_lin_model.compile(loss='mean_squared_error', optimizer="adam",metrics=['mse'])  
        return non_lin_model

class CNNa(Architecture):

    def __init__(self, seq_len, batch_size=10, validation_split=0.0, epochs=20, alphabet="UCGA", filters=50, hidden_dims=100):
        super(CNNa, self).__init__(seq_len, batch_size, validation_split, epochs, alphabet)
        self.filters = filters
        self.hidden_dims = hidden_dims
        self.architecture_name=f'CNNa_hd{self.hidden_dims}_f{self.filters}'

    def get_model(self):
        filters = self.filters 
        hidden_dims = self.hidden_dims

        model = Sequential()
        model.add(Conv1D(filters,
                         self.alphabet_len-1,  
                         padding='valid',
                         strides=1,
                         input_shape=(self.alphabet_len,self.seq_len)))

        model.add(Conv1D(filters,
                         20,
                         padding='same',
                         activation='relu',
                         strides=1))

        model.add(MaxPooling1D(1))
        model.add(Conv1D(filters,
                         self.alphabet_len-1,
                         padding='same',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error',  optimizer="adam", metrics=['mse'])
        return model

class Logistic(Architecture):
    def get_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(1, input_shape=(self.seq_len * self.alphabet_len,)))
        model.add(Activation('softmax'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        return model
    
