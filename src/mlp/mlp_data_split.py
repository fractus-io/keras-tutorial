'''

 Copyright (c) 2018 Fractus IT d.o.o. <http://fractus.io>
 
'''

# importing the libraries  
import numpy as np
import pandas as pd
import keras
import utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split 
    
def main():

    # import dataset 
    dataset = pd.read_csv('Churn_Modelling.csv')

    # prepareDate
    X, Y = utils.prepareData(dataset)
                            
    # create model        
    model = utils.getModel()
       
    # compile model   
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.fit(X, Y,  validation_split = 0.33, batch_size = 10, epochs = 100)
    
    # Split the dataset into the Training set(80%) and Test set(20%)    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    # fit the model
    model.fit(X_train, Y_train,  validation_data = (X_test, Y_test), batch_size = 10, epochs = 100)
    
    # evaluate the model
    scores = model.evaluate(X_test, X_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
if __name__ == '__main__':    
    main()

