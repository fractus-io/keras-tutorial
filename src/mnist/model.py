'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras import regularizers
from callbacks import TimeCallback

class Model:
    
    # overview of the regularization techniques https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
    
    @staticmethod
    def getModel(model_name):
    
        model = Sequential()
    
        if model_name == 'mlp_one_layer': # 
            model.add(Dense(512, activation='relu', input_shape=(784,)))        
            model.add(Dense(10, activation='softmax'))
                                     
        elif model_name == 'mlp_two_layers': # feedforward Neural Networks, also known as Deep feedforward Networks or Multi-layer Perceptrons.
            model.add(Dense(512, activation='relu', input_shape=(784,)))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            
        elif model_name == 'mlp_two_layers_dropout': # MLP with dropouts (reduce overfit whih happens with mlp_two_layers)         
            model.add(Dense(512, activation='relu', input_shape=(784,)))
            model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))
            
        elif model_name == 'conv_net':
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(10, activation='softmax'))                
                    
        elif model_name == 'conv_net_dropout':
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))                
                    
        elif model_name == 'conv_net_batch_norm':
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 )))
            model.add(MaxPooling2D((2, 2)))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(10, activation='softmax'))                
                    
        elif model_name == 'conv_net_l1':
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 ), kernel_regularizer = regularizers.l2(0.0001)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
            model.add(Flatten())
            model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l1(0.0001)))
            model.add(Dense(10, activation='softmax'))                
                            
        elif model_name == 'conv_net_l2':
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1 ), kernel_regularizer = regularizers.l2(0.0001)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
            model.add(Flatten())
            model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
            model.add(Dense(10, activation='softmax'))                
            return model

        elif model_name == 'leNet':
            model.add(Conv2D(20, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1 )))
            model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2))) 
                   
            model.add(Conv2D(50, (5, 5), padding = 'same', activation = 'relu'))
            model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))        
            
            model.add(Flatten())
            model.add(Dense(500, activation='relu'))        
            
            model.add(Dense(unique_test_labels_len, activation='softmax'))
            return model
        
        return model    

    @staticmethod
    def getCallbacks(model_name):
    
    
            # time callback
        time_callback = TimeCallback()
        
        return [time_callback]
        