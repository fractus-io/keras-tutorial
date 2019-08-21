'''

 Copyright (c) 2019 Fractus IT d.o.o. <http://fractus.io>
 
'''

# importing the libraries  
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

def main():

    # prepare data set
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
            
    # create model   
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(8, activation = 'relu', input_dim = 2))
    # Adding the second hidden layer
    model.add(Dense(4, activation = 'relu'))
    # Adding the output layer
    model.add(Dense(1, activation = 'sigmoid'))

    
    # compile model   
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # fit the model
    model.fit(X, Y, batch_size = 1, epochs = 500)
    
    # evaluate the model
    scores = model.evaluate(X, Y)

    # print accuracy
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    print(model.predict_proba(X))

    print(model.summary())


if __name__ == '__main__':    
    main()
